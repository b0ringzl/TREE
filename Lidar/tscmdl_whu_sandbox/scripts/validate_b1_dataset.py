"""Validate every exported B1 point-cloud/image pair and split contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan_path = Path(manifest["source_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    expected_count = int(manifest["summary"]["sample_count"])
    if len(records) != expected_count:
        raise ValueError(f"Manifest record count mismatch: {len(records)} != {expected_count}")

    sample_keys = [str(record["sample_key"]) for record in records]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Duplicate sample keys detected")
    point_paths = [str(record["point_path"]) for record in records]
    image_paths = [str(record["image_path"]) for record in records]
    if len(point_paths) != len(set(point_paths)) or len(image_paths) != len(set(image_paths)):
        raise ValueError("Duplicate output paths detected")

    split_histogram: dict[str, Counter[int]] = defaultdict(Counter)
    source_below_target = Counter()
    minimum_unique = 8192
    maximum_centroid_error = 0.0
    maximum_radius_error = 0.0
    minimum_crop_fraction = 1.0

    for index, record in enumerate(records, start=1):
        point_path = (root / str(record["point_path"])).resolve()
        image_path = (root / str(record["image_path"])).resolve()
        if not point_path.is_relative_to(root) or not image_path.is_relative_to(root):
            raise ValueError(f"Output path escapes dataset root: {record['sample_key']}")
        if not point_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"Missing pair for {record['sample_key']}")
        if sha256_file(point_path) != record["point_sha256"]:
            raise ValueError(f"Point hash mismatch: {record['sample_key']}")
        if sha256_file(image_path) != record["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {record['sample_key']}")

        with np.load(point_path, allow_pickle=False) as archive:
            required = {
                "points_xyz",
                "class_index",
                "benchmark_label_id",
                "centroid_xyz",
                "scale",
                "source_point_count",
                "sampled_unique_point_count",
                "sample_seed",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Missing NPZ keys for {record['sample_key']}: {sorted(missing)}")
            points = archive["points_xyz"]
            if points.shape != (8192, 3) or points.dtype != np.float32:
                raise ValueError(f"Invalid point tensor for {record['sample_key']}: {points.shape} {points.dtype}")
            if not np.isfinite(points).all():
                raise ValueError(f"Non-finite points: {record['sample_key']}")
            if int(archive["class_index"]) != int(record["class_index"]):
                raise ValueError(f"Class-index mismatch: {record['sample_key']}")
            if int(archive["benchmark_label_id"]) != int(record["benchmark_label_id"]):
                raise ValueError(f"Benchmark-label mismatch: {record['sample_key']}")
            if int(archive["source_point_count"]) != int(record["source_point_count"]):
                raise ValueError(f"Source-count mismatch: {record['sample_key']}")
            unique_count = int(archive["sampled_unique_point_count"])
            if unique_count != int(record["sampled_unique_point_count"]):
                raise ValueError(f"Unique-count mismatch: {record['sample_key']}")
            minimum_unique = min(minimum_unique, unique_count)
            centroid_error = float(np.abs(points.astype(np.float64).mean(axis=0)).max())
            radius_error = abs(float(np.linalg.norm(points, axis=1).max()) - 1.0)
            maximum_centroid_error = max(maximum_centroid_error, centroid_error)
            maximum_radius_error = max(maximum_radius_error, radius_error)
            if centroid_error > 1e-5 or radius_error > 1e-5:
                raise ValueError(f"Normalization mismatch: {record['sample_key']}")

        with Image.open(image_path) as image:
            if image.size != (768, 512) or image.mode != "RGB":
                raise ValueError(
                    f"Invalid image for {record['sample_key']}: {image.size} {image.mode}"
                )

        split = str(record["split"])
        label = int(record["benchmark_label_id"])
        split_histogram[split][label] += 1
        if int(record["source_point_count"]) < 8192:
            source_below_target[label] += 1
        minimum_crop_fraction = min(
            minimum_crop_fraction, float(record["crop_visible_fraction"])
        )
        if index % 100 == 0:
            print(f"Validated {index}/{len(records)} pairs", flush=True)

    validation_roads = set(plan["summary"]["validation_roads"])
    train_roads = {str(record["road_id"]) for record in records if record["split"] == "train"}
    val_roads = {str(record["road_id"]) for record in records if record["split"] == "val"}
    if val_roads != validation_roads or train_roads & validation_roads:
        raise ValueError("Validation-road isolation failed")
    if any(not record["is_official_test"] for record in records if record["split"] == "test"):
        raise ValueError("A test sample is not from an official reference trajectory")
    if any(record["is_official_test"] for record in records if record["split"] != "test"):
        raise ValueError("An official test sample leaked into train/validation")

    files = [path for path in root.rglob("*") if path.is_file()]
    result = {
        "status": "passed",
        "sample_count": len(records),
        "point_file_count": len(list((root / "points").rglob("*.npz"))),
        "image_file_count": len(list((root / "images").rglob("*.jpg"))),
        "preview_file_count": len(list((root / "quality_previews").glob("*.jpg"))),
        "split_histogram": {
            split: {str(label): count for label, count in sorted(hist.items())}
            for split, hist in sorted(split_histogram.items())
        },
        "validation_roads": sorted(validation_roads),
        "test_samples_are_official_references": True,
        "source_below_8192_by_label": {
            str(label): count for label, count in sorted(source_below_target.items())
        },
        "minimum_sampled_unique_point_count": minimum_unique,
        "maximum_normalized_centroid_error": maximum_centroid_error,
        "maximum_normalized_radius_error": maximum_radius_error,
        "minimum_crop_visible_fraction": minimum_crop_fraction,
        "dataset_file_count": len(files),
        "dataset_size_bytes": sum(path.stat().st_size for path in files),
    }
    if result["point_file_count"] != len(records) or result["image_file_count"] != len(records):
        raise ValueError("Unexpected point/image file count")
    if result["preview_file_count"] != manifest["summary"]["preview_count"]:
        raise ValueError("Unexpected preview file count")
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
