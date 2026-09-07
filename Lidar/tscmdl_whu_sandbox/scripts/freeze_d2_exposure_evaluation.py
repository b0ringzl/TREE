"""Freeze the user-confirmed, model-unseen D2 exposure evaluation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_REVIEW_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_target_aware_visual_confirmation_v2"
)
DEFAULT_OUTPUT_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_frozen_three_modality_evaluation_v1"
)
DEFAULT_TRAINING_ROOT = (
    DERIVED_ROOT / "d1_four_class_training_package" / "20260817_165127"
)
DEFAULT_SHARED_ROOT = DERIVED_ROOT / "c1_full_shared_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_relabel_point(source: Path, target: Path, class_index: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    points = np.asarray(payload["points_xyz"], dtype=np.float32)
    if points.shape != (8192, 3) or not np.isfinite(points).all():
        raise ValueError(f"Invalid source point tensor: {source}")
    payload["class_index"] = np.asarray(class_index, dtype=np.int64)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    review_root = args.review_root.resolve()
    training_root = args.training_root.resolve()
    shared_root = args.shared_root.resolve()
    output_root = args.output_root.resolve()
    review_manifest_path = review_root / "visualization_manifest.json"
    confirmation_path = review_root / "user_exposure_confirmation.json"
    training_manifest_path = training_root / "manifest.json"
    review = json.loads(review_manifest_path.read_text(encoding="utf-8-sig"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8-sig"))
    training = json.loads(training_manifest_path.read_text(encoding="utf-8-sig"))
    records = list(review["records"])
    confirmations = confirmation["confirmations"]
    if len(confirmations) != len(records):
        raise ValueError(
            f"D2 visual confirmation is incomplete: {len(confirmations)}/{len(records)}"
        )
    seen_keys = {str(item["sample_key"]) for item in training["records"]}
    classes = [
        {"class_index": int(item["class_index"]), "scientific_name": str(item["scientific_name"])}
        for item in sorted(review["classes"], key=lambda item: int(item["class_index"]))
    ]
    class_names = {int(item["class_index"]): str(item["scientific_name"]) for item in classes}
    frozen: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for item in records:
        key = str(item["sample_key"])
        decision = confirmations[key]
        status = str(decision["status"])
        actual_group = str(decision["actual_group"])
        if status != "accurate" or actual_group not in {"bright", "dark"}:
            exclusions.append(
                {
                    "sample_key": key,
                    "reason": "user_confirmed_normal_or_uncertain",
                    "status": status,
                    "actual_group": actual_group,
                }
            )
            continue
        if key in seen_keys:
            exclusions.append(
                {
                    "sample_key": key,
                    "reason": "already_present_in_d1_train_val_or_test_package",
                    "status": status,
                    "actual_group": actual_group,
                }
            )
            continue
        image_source = (shared_root / str(item["image_path"])).resolve()
        point_source = (shared_root / str(item["point_path"])).resolve()
        if sha256_file(image_source) != str(item["image_sha256"]):
            raise ValueError(f"Source image hash mismatch: {key}")
        if sha256_file(point_source) != str(item["point_sha256"]):
            raise ValueError(f"Source point hash mismatch: {key}")
        class_index = int(item["model_class_index"])
        point_target = output_root / "assets" / "points" / f"{key}.npz"
        atomic_relabel_point(point_source, point_target, class_index)
        point_hash = sha256_file(point_target)
        frozen.append(
            {
                "sample_key": key,
                "split": "d2_exposure_evaluation",
                "class_index": class_index,
                "scientific_name": class_names[class_index],
                "image_path": os.path.relpath(image_source, output_root),
                "point_path": os.path.relpath(point_target, output_root),
                "selection_tier": "user_confirmed_target_aware_exposure",
                "resolution_reason": "d2_target_aware_v2_user_confirmed_model_unseen",
                "source_image_path": str(image_source),
                "source_image_sha256": str(item["image_sha256"]),
                "packaged_image_sha256": str(item["image_sha256"]),
                "source_point_path": str(point_source),
                "source_point_sha256": str(item["point_sha256"]),
                "packaged_point_sha256": point_hash,
                "benchmark_label_id": int(item["benchmark_label_id"]),
                "raw_label_id": int(item["raw_label_id"]),
                "road_id": str(item["road_id"]),
                "trajectory_id": str(item["trajectory_id"]),
                "tree_id": int(item["tree_id"]),
                "source_scientific_name": str(item["source_scientific_name"]),
                "source_split_label": str(item["model_split"]),
                "automatic_quality_tier": str(item["automatic_quality_tier"]),
                "automatic_risk_score": float(item["automatic_risk_score"]),
                "automatic_risk_reasons": [
                    "excessive_bright_pixels"
                    if actual_group == "bright"
                    else "excessive_dark_pixels"
                ],
                "d2_exposure_group": actual_group,
                "d2_detector_version": str(item["d2_detector_version"]),
                "d2_exposure_rule": str(item["d2_exposure_rule"]),
                "d2_target_metrics": item["d2_target_metrics"],
                "user_confirmation": decision,
            }
        )

    if not frozen:
        raise ValueError("No model-unseen confirmed exposure samples were frozen")
    frozen.sort(key=lambda item: str(item["sample_key"]))
    class_counts = Counter(str(item["scientific_name"]) for item in frozen)
    exposure_counts = Counter(str(item["d2_exposure_group"]) for item in frozen)
    road_counts = Counter(str(item["road_id"]) for item in frozen)
    source_species_counts = Counter(str(item["source_scientific_name"]) for item in frozen)
    manifest = {
        "format_version": 2,
        "stage": "D2-frozen-model-unseen-exposure-evaluation",
        "status": "frozen",
        "generated_at": timestamp(),
        "scope": "three-modality inference on user-confirmed target-tree exposure anomalies",
        "interpretation_limit": (
            "Primary cohort excludes every sample already present in the D1 train, "
            "validation, or test package. The cohort is exposure-enriched and class-imbalanced; "
            "accuracy is descriptive and must be accompanied by class/exposure strata."
        ),
        "classes": classes,
        "summary": {
            "sample_count": len(frozen),
            "exposure_counts": dict(sorted(exposure_counts.items())),
            "class_counts": dict(sorted(class_counts.items())),
            "road_count": len(road_counts),
            "source_species_count": len(source_species_counts),
            "excluded_count": len(exclusions),
            "excluded_by_reason": dict(
                sorted(Counter(str(item["reason"]) for item in exclusions).items())
            ),
        },
        "records": frozen,
    }
    manifest_path = output_root / "manifest.json"
    atomic_json(manifest_path, manifest)
    protocol = {
        "schema_version": 1,
        "stage": "D2-frozen-three-modality-inference",
        "status": "frozen",
        "frozen_at": manifest["generated_at"],
        "no_training": True,
        "cohort_policy": "user-confirmed exposure anomaly and absent from D1 package",
        "source_bindings": {
            "visualization_manifest": {
                "path": str(review_manifest_path),
                "sha256": sha256_file(review_manifest_path),
            },
            "user_confirmation": {
                "path": str(confirmation_path),
                "sha256": sha256_file(confirmation_path),
            },
            "d1_training_manifest": {
                "path": str(training_manifest_path),
                "sha256": sha256_file(training_manifest_path),
            },
            "frozen_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
        "next_action": "run frozen image, point, and fusion checkpoints without retraining",
    }
    atomic_json(output_root / "protocol.json", protocol)
    validation = {
        "status": "passed",
        "validated_at": timestamp(),
        "all_review_samples_confirmed": len(confirmations) == len(records),
        "review_sample_count": len(records),
        "confirmed_exposure_count": sum(
            1
            for value in confirmations.values()
            if value["status"] == "accurate" and value["actual_group"] in {"bright", "dark"}
        ),
        "frozen_model_unseen_count": len(frozen),
        "frozen_keys_unique": len({item["sample_key"] for item in frozen}) == len(frozen),
        "d1_package_overlap_count": sum(item["sample_key"] in seen_keys for item in frozen),
        "exclusions": exclusions,
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_json(output_root / "validation.json", validation)
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest": str(manifest_path),
                "summary": manifest["summary"],
                "validation": str(output_root / "validation.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
