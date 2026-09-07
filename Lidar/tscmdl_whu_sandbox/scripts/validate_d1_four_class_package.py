"""Validate a built D1 four-class package without starting model training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_PACKAGE_ROOT = DERIVED_ROOT / "d1_four_class_training_package"
DEFAULT_REGISTRY = (
    DERIVED_ROOT
    / "d1_four_class_image_quality"
    / "tracked_samples"
    / "tri_modal_probe_registry.json"
)
POINT_KEYS = {
    "points_xyz",
    "class_index",
    "benchmark_label_id",
    "raw_label_id",
    "centroid_xyz",
    "scale",
    "source_point_count",
    "sampled_unique_point_count",
    "sample_seed",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_package(output_root: Path) -> Path:
    packages = sorted(
        path
        for path in output_root.glob("*")
        if path.is_dir() and (path / "package_summary.json").is_file()
    )
    if not packages:
        raise FileNotFoundError(f"No D1 package found under {output_root}")
    return packages[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--probe-registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def manifest_map(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{label} records must be a list")
    output = {str(record["sample_key"]): record for record in records}
    if len(output) != len(records):
        raise ValueError(f"Duplicate sample key in {label}")
    return output


def resolve_asset(package_root: Path, manifest_path: Path, relative: object) -> Path:
    path = (manifest_path.parent / str(relative)).resolve()
    try:
        path.relative_to(package_root)
    except ValueError as error:
        raise ValueError(f"Asset escapes package root: {relative}") from error
    return path


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    package_root = (
        args.package_root.resolve()
        if args.package_root is not None
        else latest_package(output_root)
    )
    registry_path = args.probe_registry.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    summary_path = package_root / "package_summary.json"
    clean_manifest_path = package_root / "manifest.json"
    silver_manifest_path = package_root / "silver" / "manifest.json"
    holdout_manifest_path = package_root / "holdout" / "manifest.json"
    classes_path = package_root / "classes.json"
    inventory_path = package_root / "asset_inventory.csv"
    bindings_path = package_root / "source_bindings.json"
    required = [
        summary_path,
        clean_manifest_path,
        silver_manifest_path,
        holdout_manifest_path,
        classes_path,
        inventory_path,
        bindings_path,
        package_root / "quality_reviews_for_training.json",
        package_root / "excluded_training_images.csv",
        package_root / "unreviewed_risk_silver_only.csv",
        package_root / "test_silver_inventory.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package files: {missing}")

    summary = read_json(summary_path)
    clean_manifest = read_json(clean_manifest_path)
    silver_manifest = read_json(silver_manifest_path)
    holdout_manifest = read_json(holdout_manifest_path)
    classes = json.loads(classes_path.read_text(encoding="utf-8-sig"))
    bindings = read_json(bindings_path)
    inventory = read_csv(inventory_path)
    registry = read_json(registry_path)

    if [int(item["class_index"]) for item in classes] != [0, 1, 2, 3]:
        errors.append("classes.json indices are not contiguous 0..3")
    expected_names = [
        "Cinnamomum camphora",
        "Lagerstroemia indica",
        "Magnolia grandiflora",
        "Other",
    ]
    if [str(item["scientific_name"]) for item in classes] != expected_names:
        errors.append("classes.json does not match the D1 four-class taxonomy")
    for name, payload in (
        ("clean", clean_manifest),
        ("silver", silver_manifest),
        ("holdout", holdout_manifest),
    ):
        if payload.get("image_size") != [768, 512]:
            errors.append(f"{name} manifest image_size is not 768x512")
        if payload.get("package_id") != summary.get("package_id"):
            errors.append(f"{name} manifest package_id mismatch")

    clean_map = manifest_map(clean_manifest, "clean manifest")
    silver_map = manifest_map(silver_manifest, "silver manifest")
    holdout_map = manifest_map(holdout_manifest, "holdout manifest")
    clean_train = {
        key for key, record in clean_map.items() if str(record["split"]) == "train"
    }
    silver_train = {
        key for key, record in silver_map.items() if str(record["split"]) == "train"
    }
    clean_val = {
        key for key, record in clean_map.items() if str(record["split"]) == "val"
    }
    clean_test = {
        key for key, record in clean_map.items() if str(record["split"]) == "test"
    }
    silver_val = {
        key for key, record in silver_map.items() if str(record["split"]) == "val"
    }
    silver_test = {
        key for key, record in silver_map.items() if str(record["split"]) == "test"
    }
    holdout_keys = set(holdout_map)
    expected_counts = summary["counts"]
    observed_counts = {
        "train_clean": len(clean_train),
        "train_silver": len(silver_train),
        "train_silver_only": len(silver_train - clean_train),
        "exposure_holdout": len(holdout_keys),
        "formal_val": len(clean_val),
        "formal_test": len(clean_test),
        "formal_evaluation_total": len(clean_val | clean_test),
        "packaged_unique_assets": len(inventory),
    }
    for name, observed in observed_counts.items():
        expected = int(expected_counts[name])
        if observed != expected:
            errors.append(f"{name} count mismatch: {observed} != {expected}")

    if not clean_train <= silver_train:
        errors.append("clean training set is not a subset of silver training set")
    if clean_val != silver_val or clean_test != silver_test:
        errors.append("clean and silver formal evaluation sets differ")
    if (clean_train | silver_train) & (clean_val | clean_test):
        errors.append("training and formal evaluation sample keys overlap")
    if holdout_keys & (silver_train | clean_val | clean_test):
        errors.append("exposure holdout leaks into training or formal evaluation")
    if {str(record["split"]) for record in holdout_map.values()} != {
        "exposure_holdout"
    }:
        errors.append("holdout manifest contains a non-holdout split")
    if {str(record["split"]) for record in clean_map.values()} != {
        "train",
        "val",
        "test",
    }:
        errors.append("clean manifest split set is not train/val/test")

    cohort = next(
        (
            item
            for item in registry.get("cohorts", [])
            if item.get("cohort_id") == "EXPOSURE_NEGATIVE_TRANSFER_V1"
        ),
        None,
    )
    if cohort is None:
        errors.append("exposure cohort missing from probe registry")
    elif holdout_keys != {str(key) for key in cohort.get("sample_keys", [])}:
        errors.append("package holdout does not match the registered exposure cohort")

    for split_name, keys, records in (
        ("train_clean", clean_train, clean_map),
        ("train_silver", silver_train, silver_map),
        ("val", clean_val, clean_map),
        ("test", clean_test, clean_map),
    ):
        counts = Counter(int(records[key]["class_index"]) for key in keys)
        if set(counts) != {0, 1, 2, 3}:
            errors.append(f"{split_name} does not cover all four classes: {dict(counts)}")
    holdout_class_indices = {
        int(holdout_map[key]["class_index"]) for key in holdout_keys
    }
    if holdout_class_indices != {0, 1, 2}:
        errors.append(
            "exposure holdout must cover the three shared target species and exclude Other: "
            f"{sorted(holdout_class_indices)}"
        )

    if len(inventory) != len({row["sample_key"] for row in inventory}):
        errors.append("asset_inventory.csv has duplicate sample keys")
    inventory_map = {row["sample_key"]: row for row in inventory}
    union_keys = silver_train | clean_val | clean_test | holdout_keys
    if set(inventory_map) != union_keys:
        errors.append("asset inventory keys do not equal silver/evaluation/holdout union")

    for binding_name, binding in bindings.items():
        source = PROJECT_ROOT / str(binding["path"])
        if not source.is_file():
            errors.append(f"source binding missing: {binding_name}")
        elif sha256_file(source) != str(binding["sha256"]):
            errors.append(f"source binding hash changed: {binding_name}")

    all_manifest_records: dict[str, dict[str, Any]] = {}
    for key in union_keys:
        candidates = [
            mapping[key]
            for mapping in (silver_map, holdout_map)
            if key in mapping
        ]
        all_manifest_records[key] = candidates[0]

    validated_bytes = 0
    hardlink_failures = 0
    for index, key in enumerate(sorted(union_keys), start=1):
        record = all_manifest_records[key]
        row = inventory_map[key]
        owning_manifest = holdout_manifest_path if key in holdout_keys else silver_manifest_path
        image_path = resolve_asset(package_root, owning_manifest, record["image_path"])
        point_path = resolve_asset(package_root, owning_manifest, record["point_path"])
        source_image = PROJECT_ROOT / row["image_source_path"]
        source_point = PROJECT_ROOT / row["point_source_path"]
        for path, kind in (
            (image_path, "packaged image"),
            (point_path, "packaged point"),
            (source_image, "source image"),
            (source_point, "source point"),
        ):
            if not path.is_file():
                errors.append(f"{kind} missing for {key}: {path}")
        if any(not path.is_file() for path in (image_path, point_path, source_image, source_point)):
            continue
        image_hash = sha256_file(image_path)
        point_hash = sha256_file(point_path)
        source_point_hash = sha256_file(source_point)
        validated_bytes += image_path.stat().st_size + point_path.stat().st_size
        if image_hash != row["image_sha256"] or image_hash != record["packaged_image_sha256"]:
            errors.append(f"packaged image hash mismatch: {key}")
        if point_hash != row["point_sha256"] or point_hash != record["packaged_point_sha256"]:
            errors.append(f"packaged point hash mismatch: {key}")
        if source_point_hash != row["point_source_sha256"] or source_point_hash != record["source_point_sha256"]:
            errors.append(f"source point hash mismatch: {key}")
        source_stat = os.stat(source_image)
        image_stat = os.stat(image_path)
        if (source_stat.st_dev, source_stat.st_ino) != (image_stat.st_dev, image_stat.st_ino):
            hardlink_failures += 1
        try:
            with Image.open(image_path) as image:
                image.load()
                if image.size != (768, 512):
                    errors.append(f"unexpected image size for {key}: {image.size}")
        except OSError as error:
            errors.append(f"image decode failed for {key}: {error}")
        try:
            with np.load(point_path, allow_pickle=False) as archive:
                if set(archive.files) != POINT_KEYS:
                    errors.append(f"unexpected point keys for {key}")
                    continue
                points = archive["points_xyz"]
                label = int(archive["class_index"])
                benchmark_label = int(archive["benchmark_label_id"])
            if points.shape != (8192, 3) or points.dtype != np.float32:
                errors.append(f"unexpected point tensor for {key}: {points.shape} {points.dtype}")
            elif not np.isfinite(points).all():
                errors.append(f"non-finite point tensor for {key}")
            if label != int(record["class_index"]):
                errors.append(f"four-class point label mismatch for {key}")
            if benchmark_label != int(record["benchmark_label_id"]):
                errors.append(f"benchmark point label mismatch for {key}")
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"point decode failed for {key}: {error}")
        if index % 500 == 0 or index == len(union_keys):
            print(f"validated assets {index}/{len(union_keys)}", flush=True)

    if hardlink_failures:
        errors.append(f"{hardlink_failures} packaged images are not hard links to their sources")
    forbidden_training_artifacts = sorted(
        str(path.relative_to(package_root))
        for pattern in ("*.pt", "*.pth", "train.log", "run_state.json")
        for path in package_root.rglob(pattern)
    )
    if forbidden_training_artifacts:
        errors.append(f"training artifacts unexpectedly exist: {forbidden_training_artifacts[:5]}")

    normal_control_status = str(cohort.get("normal_exposure_control_status", "")) if cohort else ""
    if normal_control_status != "selected":
        warnings.append("normal-exposure matched controls remain pending selection")
    status = "failed" if errors else "passed_with_warnings" if warnings else "passed"
    result = {
        "stage": "D1-four-class-package-validation",
        "status": status,
        "validated_at": now_iso(),
        "package_root": str(package_root),
        "package_id": summary.get("package_id"),
        "observed_counts": observed_counts,
        "class_counts": {
            "train_clean": dict(sorted(Counter(clean_map[key]["scientific_name"] for key in clean_train).items())),
            "train_silver": dict(sorted(Counter(silver_map[key]["scientific_name"] for key in silver_train).items())),
            "val": dict(sorted(Counter(clean_map[key]["scientific_name"] for key in clean_val).items())),
            "test": dict(sorted(Counter(clean_map[key]["scientific_name"] for key in clean_test).items())),
            "holdout": dict(sorted(Counter(holdout_map[key]["scientific_name"] for key in holdout_keys).items())),
        },
        "validated_unique_assets": len(union_keys),
        "validated_bytes": validated_bytes,
        "image_hardlinks_verified": len(union_keys) - hardlink_failures,
        "source_bindings_verified": len(bindings),
        "four_class_point_labels_verified": len(union_keys),
        "training_artifacts_found": forbidden_training_artifacts,
        "training_started": False,
        "errors": errors,
        "warnings": warnings,
    }
    write_json(package_root / "validation.json", result)
    summary["status"] = status
    summary["validated_at"] = result["validated_at"]
    summary["validation_path"] = "validation.json"
    atomic_summary = Path(f"{summary_path}.tmp")
    atomic_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    atomic_summary.replace(summary_path)
    report = [
        "# D1 四分类正式数据包验证报告",
        "",
        f"- 验证时间：{result['validated_at']}",
        f"- 状态：**{status}**",
        f"- 唯一图像/点云对：{len(union_keys)}",
        f"- clean 训练：{len(clean_train)}",
        f"- silver 训练：{len(silver_train)}",
        f"- 曝光留出：{len(holdout_keys)}",
        f"- 正式验证/测试：{len(clean_val)}/{len(clean_test)}",
        f"- 图像硬链接验证：{len(union_keys) - hardlink_failures}/{len(union_keys)}",
        f"- 四分类点云标签验证：{len(union_keys)}/{len(union_keys)}",
        f"- 源清单绑定验证：{len(bindings)}",
        "- 训练状态：**未启动**",
        "",
        "## 结论",
        "",
        "数据划分、人工审核决策、替代影像、曝光留出、源哈希、图像解码和四分类点云标签均已逐项检查。",
    ]
    if warnings:
        report.extend(["", "## 待办警告", "", *[f"- {item}" for item in warnings]])
    if errors:
        report.extend(["", "## 错误", "", *[f"- {item}" for item in errors]])
    report.extend(
        [
            "",
            "科研灵感：通过同一留出集合锁定图像、点云与融合三条预测支路，可将曝光负迁移从个案观察提升为逐样本配对证据。",
            "",
        ]
    )
    (package_root / "D1四分类正式数据包验证报告.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
