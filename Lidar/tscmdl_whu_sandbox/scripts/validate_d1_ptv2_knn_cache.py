"""Validate the D1 four-class PTv2 memory-mapped kNN cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-samples", type=int, default=32)
    parser.add_argument("--expected-class-count", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def close_memmap(array: np.ndarray) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def main() -> None:
    args = parse_args()
    if args.chunk_samples <= 0 or args.expected_class_count < 2:
        raise ValueError("Invalid validation arguments")
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
    if len(classes) != args.expected_class_count:
        errors.append(
            f"class count is {len(classes)}, expected {args.expected_class_count}"
        )
    if [int(item["class_index"]) for item in classes] != list(range(len(classes))):
        errors.append("class indices are not contiguous")
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
            (item for item in manifest["records"] if str(item["split"]) == split),
            key=lambda item: str(item["sample_key"]),
        )
        for split in SPLITS
    }
    total_samples = sum(len(items) for items in records_by_split.values())
    if int(index.get("geometry_verified_samples", -1)) != total_samples:
        errors.append("not all point geometries were verified during construction")
    split_results: dict[str, Any] = {}
    total_bytes = 0
    for split in SPLITS:
        entry = index.get("splits", {}).get(split)
        if entry is None:
            errors.append(f"missing index entry for {split}")
            continue
        expected_keys = tuple(
            str(item["sample_key"]) for item in records_by_split[split]
        )
        if tuple(str(value) for value in entry.get("sample_keys", ())) != expected_keys:
            errors.append(f"sample order mismatch for {split}")
        path = cache_dir / str(entry.get("path", ""))
        if not path.is_file():
            errors.append(f"missing cache file for {split}")
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        minimum = 8192
        maximum = -1
        expected_shape = (len(expected_keys), 8192, 8)
        try:
            if array.shape != expected_shape:
                errors.append(f"shape mismatch for {split}: {array.shape}")
            if array.dtype != np.int32:
                errors.append(f"dtype mismatch for {split}: {array.dtype}")
            for begin in range(0, len(array), args.chunk_samples):
                values = np.asarray(array[begin : begin + args.chunk_samples])
                if values.size:
                    minimum = min(minimum, int(values.min()))
                    maximum = max(maximum, int(values.max()))
            if minimum < 0 or maximum >= 8192:
                errors.append(f"index range outside [0,8192) for {split}")
        finally:
            close_memmap(array)
        actual_hash = sha256_file(path)
        if actual_hash != str(entry.get("sha256")):
            errors.append(f"cache SHA-256 mismatch for {split}")
        if path.stat().st_size != int(entry.get("bytes", -1)):
            errors.append(f"cache byte count mismatch for {split}")
        total_bytes += path.stat().st_size
        split_results[split] = {
            "sample_count": len(expected_keys),
            "shape": list(expected_shape),
            "dtype": "int32",
            "minimum_index": minimum,
            "maximum_index": maximum,
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
        }

    result = {
        "format_version": 1,
        "stage": "D1-four-class-PTv2-cache-validation",
        "status": "passed" if not errors else "failed",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "class_count": len(classes),
        "sample_count": total_samples,
        "geometry_verified_samples": int(index.get("geometry_verified_samples", 0)),
        "cache_bytes": total_bytes,
        "index_sha256": sha256_file(index_path),
        "split_results": split_results,
        "elapsed_seconds": time.perf_counter() - started,
        "errors": errors,
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
