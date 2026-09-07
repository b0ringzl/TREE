"""Validate the complete C2a PTv2 memory-mapped kNN cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-samples", type=int, default=32)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def close_memmap(array: np.ndarray) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def main() -> None:
    args = parse_args()
    if args.chunk_samples <= 0:
        raise ValueError("chunk-samples must be positive")
    started = time.perf_counter()
    dataset_root = args.dataset_root.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest_path = dataset_root / "manifest.json"
    classes_path = dataset_root / "classes.json"
    index_path = cache_dir / "index.json"
    checkpoint_path = cache_dir / "checkpoint.json"
    for path in (manifest_path, classes_path, index_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if len(classes) != 19:
        errors.append(f"class count is {len(classes)}, expected 19")
    if index.get("schema_version") != 2:
        errors.append("index schema_version is not 2")
    if index.get("cache_type") != "ptv2_memmap_knn":
        errors.append("index cache_type is not ptv2_memmap_knn")
    if int(index.get("point_count", 0)) != 8192:
        errors.append("index point_count is not 8192")
    if int(index.get("neighbours", 0)) != 8:
        errors.append("index neighbours is not 8")
    if index.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("manifest SHA-256 mismatch")
    if index.get("classes_sha256") != sha256_file(classes_path):
        errors.append("classes SHA-256 mismatch")
    if checkpoint.get("status") != "complete":
        errors.append("checkpoint status is not complete")

    records_by_split = {
        split: sorted(
            (
                record
                for record in manifest["records"]
                if str(record["split"]) == split
            ),
            key=lambda record: str(record["sample_key"]),
        )
        for split in SPLITS
    }
    split_results: dict[str, object] = {}
    total_samples = 0
    total_bytes = 0
    for split in SPLITS:
        entry = index.get("splits", {}).get(split)
        if entry is None:
            errors.append(f"missing index entry for {split}")
            continue
        expected_keys = tuple(
            str(record["sample_key"]) for record in records_by_split[split]
        )
        actual_keys = tuple(str(value) for value in entry.get("sample_keys", ()))
        if actual_keys != expected_keys:
            errors.append(f"sample order mismatch for {split}")
        path = cache_dir / str(entry["path"])
        if not path.is_file():
            errors.append(f"missing cache file for {split}: {path}")
            continue

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            expected_shape = (len(expected_keys), 8192, 8)
            if array.shape != expected_shape:
                errors.append(
                    f"shape mismatch for {split}: {array.shape} != {expected_shape}"
                )
            if array.dtype != np.int32:
                errors.append(f"dtype mismatch for {split}: {array.dtype}")
            minimum = 8192
            maximum = -1
            for begin in range(0, len(array), args.chunk_samples):
                values = np.asarray(
                    array[begin : begin + args.chunk_samples],
                    dtype=np.int32,
                )
                minimum = min(minimum, int(values.min()))
                maximum = max(maximum, int(values.max()))
            if minimum < 0 or maximum >= 8192:
                errors.append(
                    f"index range outside [0, 8192) for {split}: "
                    f"{minimum}..{maximum}"
                )
        finally:
            close_memmap(array)

        actual_sha256 = sha256_file(path)
        actual_bytes = path.stat().st_size
        if actual_sha256 != entry.get("sha256"):
            errors.append(f"cache SHA-256 mismatch for {split}")
        if actual_bytes != int(entry.get("bytes", -1)):
            errors.append(f"cache byte count mismatch for {split}")
        total_samples += len(expected_keys)
        total_bytes += actual_bytes
        split_results[split] = {
            "sample_count": len(expected_keys),
            "shape": list(expected_shape),
            "dtype": "int32",
            "minimum_index": minimum,
            "maximum_index": maximum,
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }
        print(
            f"{split}: {len(expected_keys)} samples, "
            f"indices={minimum}..{maximum}, hash=passed",
            flush=True,
        )

    events = checkpoint.get("run_events", [])
    partial_events = sum(event.get("status") == "partial" for event in events)
    complete_events = sum(event.get("status") == "complete" for event in events)
    if partial_events < 1 or complete_events < 1:
        errors.append("controlled partial-run plus resume evidence is incomplete")
    expected_total = int(manifest["summary"]["sample_count"])
    if total_samples != expected_total:
        errors.append(
            f"total sample count mismatch: {total_samples} != {expected_total}"
        )

    result = {
        "stage": "C2a",
        "check": "full PTv2 kNN cache",
        "status": "passed" if not errors else "failed",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "index_sha256": sha256_file(index_path),
        "sample_count": total_samples,
        "cache_bytes": total_bytes,
        "split_results": split_results,
        "resume_evidence": {
            "run_count": len(events),
            "partial_run_count": partial_events,
            "complete_run_count": complete_events,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "errors": errors,
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
