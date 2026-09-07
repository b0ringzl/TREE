"""Build a D1-bound PTv2 kNN cache by verified reindexing of the C2a cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
POINT_COUNT = 8192
NEIGHBOURS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-chunk-samples", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=128)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_with_retry(temporary: Path, destination: Path) -> None:
    delay_seconds = 0.01
    deadline = time.monotonic() + 10.0
    while True:
        try:
            temporary.replace(destination)
            return
        except OSError as error:
            retryable = os.name == "nt" and getattr(error, "winerror", None) in {5, 32}
            if not retryable or time.monotonic() >= deadline:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.6, 0.5)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    replace_with_retry(temporary, path)


def close_memmap(array: np.ndarray) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    args = parse_args()
    if args.copy_chunk_samples <= 0 or args.progress_every <= 0:
        raise ValueError("copy/progress chunk sizes must be positive")
    started = time.perf_counter()
    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    source_dataset_root = args.source_dataset_root.resolve()
    source_cache = args.source_cache.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = dataset_root / "manifest.json"
    dataset_classes_path = dataset_root / "classes.json"
    source_manifest_path = source_dataset_root / "manifest.json"
    source_classes_path = source_dataset_root / "classes.json"
    source_index_path = source_cache / "index.json"
    for path in (
        dataset_manifest_path,
        dataset_classes_path,
        source_manifest_path,
        source_classes_path,
        source_index_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    classes = json.loads(dataset_classes_path.read_text(encoding="utf-8"))
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    if source_index.get("schema_version") != 2:
        raise ValueError("Source cache schema is not version 2")
    if source_index.get("cache_type") != "ptv2_memmap_knn":
        raise ValueError("Source cache type is not PTv2 memmap kNN")
    if int(source_index.get("point_count", 0)) != POINT_COUNT:
        raise ValueError("Source cache point count is not 8192")
    if int(source_index.get("neighbours", 0)) != NEIGHBOURS:
        raise ValueError("Source cache neighbour count is not 8")
    if source_index["manifest_sha256"] != sha256_file(source_manifest_path):
        raise ValueError("Source cache no longer matches the C1 manifest")
    if source_index["classes_sha256"] != sha256_file(source_classes_path):
        raise ValueError("Source cache no longer matches the C1 classes")

    records_by_split = {
        split: sorted(
            (item for item in manifest["records"] if str(item["split"]) == split),
            key=lambda item: str(item["sample_key"]),
        )
        for split in SPLITS
    }
    split_sizes = {split: len(records_by_split[split]) for split in SPLITS}
    expected_sizes = manifest["summary"].get("split_sizes")
    if expected_sizes is None:
        expected_sizes = manifest["summary"].get("split_histogram")
    if expected_sizes is None or any(
        split_sizes[split] != int(expected_sizes[split]) for split in SPLITS
    ):
        raise ValueError("D1 manifest split sizes are inconsistent")
    total_samples = sum(split_sizes.values())
    required_bytes = total_samples * POINT_COUNT * NEIGHBOURS * 4
    if shutil.disk_usage(output_dir).free < required_bytes + 512 * 1024**2:
        raise OSError("Insufficient disk space for the D1 kNN cache plus safety margin")

    source_locations: dict[str, tuple[str, int]] = {}
    source_arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        entry = source_index["splits"][split]
        keys = [str(value) for value in entry["sample_keys"]]
        for index, key in enumerate(keys):
            if key in source_locations:
                raise ValueError(f"Duplicate source-cache sample key: {key}")
            source_locations[key] = (split, index)
        array_path = source_cache / str(entry["path"])
        if not array_path.is_file() or sha256_file(array_path) != str(entry["sha256"]):
            raise ValueError(f"Source cache file hash mismatch: {split}")
        source_arrays[split] = np.load(array_path, mmap_mode="r", allow_pickle=False)

    missing = [
        str(item["sample_key"])
        for split in SPLITS
        for item in records_by_split[split]
        if str(item["sample_key"]) not in source_locations
    ]
    split_mismatches = [
        str(item["sample_key"])
        for split in SPLITS
        for item in records_by_split[split]
        if source_locations.get(str(item["sample_key"]), (None, -1))[0] != split
    ]
    if missing or split_mismatches:
        raise ValueError(
            f"D1/source cache key mismatch: missing={len(missing)}, "
            f"split_mismatch={len(split_mismatches)}"
        )

    progress_path = output_dir / "progress.json"
    atomic_json(
        progress_path,
        {
            "status": "verifying_geometry",
            "updated_at": now(),
            "completed_samples": 0,
            "total_samples": total_samples,
        },
    )
    verified = 0
    for split in SPLITS:
        for item in records_by_split[split]:
            sample_key = str(item["sample_key"])
            packaged_path = (dataset_root / str(item["point_path"])).resolve()
            source_path = resolve_project_path(project_root, str(item["source_point_path"]))
            if sha256_file(packaged_path) != str(item["packaged_point_sha256"]):
                raise ValueError(f"Packaged point hash mismatch: {sample_key}")
            if sha256_file(source_path) != str(item["source_point_sha256"]):
                raise ValueError(f"Source point hash mismatch: {sample_key}")
            with np.load(packaged_path, allow_pickle=False) as packaged, np.load(
                source_path, allow_pickle=False
            ) as source:
                packaged_points = packaged["points_xyz"]
                source_points = source["points_xyz"]
                if (
                    packaged_points.shape != (POINT_COUNT, 3)
                    or packaged_points.dtype != np.float32
                    or not np.array_equal(packaged_points, source_points)
                ):
                    raise ValueError(f"Point geometry changed for {sample_key}")
            verified += 1
            if verified % args.progress_every == 0 or verified == total_samples:
                atomic_json(
                    progress_path,
                    {
                        "status": "verifying_geometry",
                        "updated_at": now(),
                        "completed_samples": verified,
                        "total_samples": total_samples,
                    },
                )
                print(
                    f"geometry [{verified}/{total_samples}] verified",
                    flush=True,
                )

    split_entries: dict[str, Any] = {}
    copied = 0
    for split in SPLITS:
        records = records_by_split[split]
        source_indices = np.asarray(
            [source_locations[str(item["sample_key"])][1] for item in records],
            dtype=np.int64,
        )
        final_path = output_dir / f"{split}_reference_index.npy"
        temporary_path = output_dir / f".{split}_reference_index.building.npy"
        destination = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.int32,
            shape=(len(records), POINT_COUNT, NEIGHBOURS),
        )
        source_array = source_arrays[split]
        for begin in range(0, len(records), args.copy_chunk_samples):
            end = min(begin + args.copy_chunk_samples, len(records))
            destination[begin:end] = np.asarray(
                source_array[source_indices[begin:end]], dtype=np.int32
            )
            copied += end - begin
            if copied % args.progress_every < args.copy_chunk_samples or copied == total_samples:
                destination.flush()
                atomic_json(
                    progress_path,
                    {
                        "status": "copying_cache",
                        "updated_at": now(),
                        "current_split": split,
                        "completed_samples": copied,
                        "total_samples": total_samples,
                    },
                )
                print(f"cache [{copied}/{total_samples}] copied", flush=True)
        destination.flush()
        close_memmap(destination)
        replace_with_retry(temporary_path, final_path)
        keys = tuple(str(item["sample_key"]) for item in records)
        split_entries[split] = {
            "path": final_path.name,
            "shape": [len(records), POINT_COUNT, NEIGHBOURS],
            "dtype": "int32",
            "sample_keys": list(keys),
            "sample_keys_sha256": hashlib.sha256(
                "\n".join(keys).encode("utf-8")
            ).hexdigest(),
            "bytes": final_path.stat().st_size,
            "sha256": sha256_file(final_path),
        }

    for array in source_arrays.values():
        close_memmap(array)
    index = {
        "schema_version": 2,
        "cache_type": "ptv2_memmap_knn",
        "created_at": now(),
        "dataset_root": str(dataset_root),
        "manifest_sha256": sha256_file(dataset_manifest_path),
        "classes_sha256": sha256_file(dataset_classes_path),
        "point_count": POINT_COUNT,
        "neighbours": NEIGHBOURS,
        "distance": "exact Euclidean kNN within each tree instance",
        "split_sizes": split_sizes,
        "splits": split_entries,
        "construction": "verified sample_key reindex from accepted C2a cache",
        "source_cache": {
            "path": str(source_cache),
            "index_sha256": sha256_file(source_index_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_classes_sha256": sha256_file(source_classes_path),
        },
        "geometry_verified_samples": verified,
        "elapsed_seconds": time.perf_counter() - started,
    }
    index_path = output_dir / "index.json"
    atomic_json(index_path, index)
    checkpoint = {
        "schema_version": 1,
        "stage": "D1-four-class-PTv2-cache-view",
        "status": "complete",
        "completed_at": now(),
        "sample_count": total_samples,
        "geometry_verified_samples": verified,
        "index_sha256": sha256_file(index_path),
    }
    atomic_json(output_dir / "checkpoint.json", checkpoint)
    atomic_json(
        progress_path,
        {
            "status": "complete",
            "updated_at": now(),
            "completed_samples": total_samples,
            "total_samples": total_samples,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps(checkpoint, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
