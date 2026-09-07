"""Prepare and freeze model-unseen, target-normal controls for D2 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_CASE_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_frozen_three_modality_evaluation_v1"
)
DEFAULT_REVIEW_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_target_aware_visual_confirmation_v2"
)
DEFAULT_OUTPUT_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_matched_normal_controls_v1"
)
DEFAULT_CANDIDATE = DERIVED_ROOT / "d1_four_class_image_quality" / "candidate_manifest.json"
DEFAULT_TRAINING = (
    DERIVED_ROOT / "d1_four_class_training_package" / "20260817_165127" / "manifest.json"
)
DEFAULT_SHARED = DERIVED_ROOT / "c1_full_shared_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "finalize"), required=True)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED)
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
        raise ValueError(f"Invalid point tensor: {source}")
    payload["class_index"] = np.asarray(class_index, dtype=np.int64)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(target)


def automatic_normal(record: dict[str, Any]) -> bool:
    metrics = record["automatic_metrics"]
    return (
        str(record["automatic_quality_tier"]) == "passed"
        and not record.get("automatic_risk_reasons")
        and 50.0 <= float(metrics["mean_luminance"]) <= 205.0
        and float(metrics["dark_ratio"]) <= 0.05
        and float(metrics["bright_ratio"]) <= 0.20
        and float(metrics["edge_energy"]) >= 12.0
    )


def target_normal(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["visible_projected_point_fraction"]) >= 0.95
        and 60.0 <= float(metrics["target_r5_mean_luminance"]) <= 190.0
        and float(metrics["target_r5_dark_ratio"]) <= 0.10
        and float(metrics["target_r5_bright_ratio"]) <= 0.15
        and float(metrics["target_r5_visible_ratio"]) >= 0.75
        and float(metrics["target_r5_edge_energy"]) >= 12.0
    )


def match_tier(case: dict[str, Any], control: dict[str, Any]) -> int:
    same_species = str(case["source_scientific_name"]) == str(control["source_scientific_name"])
    same_road = str(case["road_id"]) == str(control["road_id"])
    same_class = str(case["scientific_name"]) == str(control["model_class_name"])
    if same_species and same_road:
        return 0
    if same_species:
        return 1
    if same_class and same_road:
        return 2
    if same_class:
        return 3
    return 99


def geometry_cost(case: dict[str, Any], control: dict[str, Any]) -> float:
    point_cost = abs(
        math.log1p(float(case.get("source_point_count", 0)))
        - math.log1p(float(control.get("source_point_count", 0)))
    )
    distance_cost = abs(float(case["camera_distance_m"]) - float(control["camera_distance_m"])) / 10.0
    crop_cost = abs(float(case.get("crop_visible_fraction", 1.0)) - float(control.get("crop_visible_fraction", 1.0)))
    return point_cost + distance_cost + crop_cost


def prepare(args: argparse.Namespace) -> None:
    candidate_path = args.candidate_manifest.resolve()
    training_path = args.training_manifest.resolve()
    case_path = args.case_root.resolve() / "manifest.json"
    review_path = args.review_root.resolve() / "visualization_manifest.json"
    output_root = args.output_root.resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    training = json.loads(training_path.read_text(encoding="utf-8-sig"))
    cases = json.loads(case_path.read_text(encoding="utf-8-sig"))
    review = json.loads(review_path.read_text(encoding="utf-8-sig"))
    seen = {str(item["sample_key"]) for item in training["records"]}
    excluded = {str(item["sample_key"]) for item in cases["records"]}
    excluded.update(str(item["sample_key"]) for item in review["records"])
    pool = [
        dict(item)
        for item in candidate["records"]
        if str(item["sample_key"]) not in seen
        and str(item["sample_key"]) not in excluded
        and automatic_normal(item)
    ]
    if not pool:
        raise ValueError("No strict automatic-normal model-unseen controls")
    manifest = {
        "format_version": 1,
        "stage": "D2c-normal-control-target-metric-candidates",
        "status": "prepared",
        "generated_at": timestamp(),
        "classes": candidate["classes"],
        "summary": {
            "sample_count": len(pool),
            "model_unseen": True,
            "automatic_normal_policy": {
                "quality_tier": "passed",
                "risk_reasons": "empty",
                "mean_luminance": [50.0, 205.0],
                "dark_ratio_max": 0.05,
                "bright_ratio_max": 0.20,
                "edge_energy_min": 12.0,
            },
            "class_counts": dict(sorted(Counter(str(item["model_class_name"]) for item in pool).items())),
        },
        "records": sorted(pool, key=lambda item: str(item["sample_key"])),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "normal_candidate_manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        output_root / "prepare_validation.json",
        {
            "status": "passed",
            "generated_at": manifest["generated_at"],
            "case_count": len(cases["records"]),
            "candidate_count": len(pool),
            "candidate_keys_unique": len({item["sample_key"] for item in pool}) == len(pool),
            "d1_package_overlap_count": sum(item["sample_key"] in seen for item in pool),
            "d2_review_overlap_count": sum(item["sample_key"] in excluded for item in pool),
            "source_bindings": {
                "candidate": sha256_file(candidate_path),
                "training": sha256_file(training_path),
                "cases": sha256_file(case_path),
                "review": sha256_file(review_path),
            },
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(json.dumps({"status": "prepared", "candidate_count": len(pool), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    shared_root = args.shared_root.resolve()
    case_path = args.case_root.resolve() / "manifest.json"
    review_path = args.review_root.resolve() / "visualization_manifest.json"
    candidate_path = output_root / "normal_candidate_manifest.json"
    metrics_path = output_root / "target_metrics" / "target_exposure_metrics.json"
    cases = json.loads(case_path.read_text(encoding="utf-8-sig"))
    review = json.loads(review_path.read_text(encoding="utf-8-sig"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    metrics = {str(item["sample_key"]): item for item in metrics_payload["rows"]}
    normal_pool = []
    for item in candidates["records"]:
        value = metrics[str(item["sample_key"])]
        if target_normal(value):
            normal_pool.append({**item, "d2_target_metrics": value})
    review_by_key = {str(item["sample_key"]): item for item in review["records"]}
    case_records = [
        {**review_by_key[str(item["sample_key"])], **item}
        for item in cases["records"]
    ]
    if len(normal_pool) < len(case_records):
        raise ValueError(f"Insufficient target-normal controls: {len(normal_pool)}/{len(case_records)}")
    cost = np.full((len(case_records), len(normal_pool)), 1.0e12, dtype=np.float64)
    tier_matrix = np.full(cost.shape, 99, dtype=np.int64)
    for row_index, case in enumerate(case_records):
        for column_index, control in enumerate(normal_pool):
            tier = match_tier(case, control)
            tier_matrix[row_index, column_index] = tier
            if tier < 99:
                cost[row_index, column_index] = tier * 1.0e6 + geometry_cost(case, control)
    row_indices, column_indices = linear_sum_assignment(cost)
    if len(row_indices) != len(case_records) or np.any(cost[row_indices, column_indices] >= 1.0e12):
        raise ValueError("Could not assign a unique same-class normal control to every case")
    pairs = []
    frozen = []
    classes = cases["classes"]
    for pair_index, (row_index, column_index) in enumerate(zip(row_indices, column_indices, strict=True), start=1):
        case = case_records[int(row_index)]
        control = normal_pool[int(column_index)]
        key = str(control["sample_key"])
        class_index = int(control["model_class_index"])
        image_source = (shared_root / str(control["image_path"])).resolve()
        point_source = (shared_root / str(control["point_path"])).resolve()
        if sha256_file(image_source) != str(control["image_sha256"]):
            raise ValueError(f"Image hash mismatch: {key}")
        if sha256_file(point_source) != str(control["point_sha256"]):
            raise ValueError(f"Point hash mismatch: {key}")
        point_target = output_root / "assets" / "points" / f"{key}.npz"
        atomic_relabel_point(point_source, point_target, class_index)
        tier = int(tier_matrix[int(row_index), int(column_index)])
        pair = {
            "pair_id": pair_index,
            "case_sample_key": str(case["sample_key"]),
            "control_sample_key": key,
            "case_exposure_group": str(case["d2_exposure_group"]),
            "model_class_name": str(case["scientific_name"]),
            "case_source_species": str(case["source_scientific_name"]),
            "control_source_species": str(control["source_scientific_name"]),
            "case_road_id": str(case["road_id"]),
            "control_road_id": str(control["road_id"]),
            "match_tier": tier,
            "match_tier_name": (
                "same_source_species_same_road"
                if tier == 0
                else "same_source_species_other_road"
                if tier == 1
                else "same_model_class_same_road"
                if tier == 2
                else "same_model_class_other_road"
            ),
            "geometry_cost": geometry_cost(case, control),
        }
        pairs.append(pair)
        frozen.append(
            {
                "sample_key": key,
                "split": "d2_normal_control",
                "class_index": class_index,
                "scientific_name": str(control["model_class_name"]),
                "image_path": os.path.relpath(image_source, output_root),
                "point_path": os.path.relpath(point_target, output_root),
                "selection_tier": "target_normal_model_unseen_matched_control",
                "resolution_reason": pair["match_tier_name"],
                "source_image_path": str(image_source),
                "source_image_sha256": str(control["image_sha256"]),
                "packaged_image_sha256": str(control["image_sha256"]),
                "source_point_path": str(point_source),
                "source_point_sha256": str(control["point_sha256"]),
                "packaged_point_sha256": sha256_file(point_target),
                "benchmark_label_id": int(control["benchmark_label_id"]),
                "raw_label_id": int(control["raw_label_id"]),
                "road_id": str(control["road_id"]),
                "trajectory_id": str(control["trajectory_id"]),
                "tree_id": int(control["tree_id"]),
                "source_point_count": int(control["source_point_count"]),
                "source_scientific_name": str(control["source_scientific_name"]),
                "source_split_label": str(control["model_split"]),
                "camera_distance_m": float(control["camera_distance_m"]),
                "automatic_quality_tier": str(control["automatic_quality_tier"]),
                "automatic_risk_score": float(control["automatic_risk_score"]),
                "automatic_risk_reasons": [],
                "d2_exposure_group": "normal",
                "d2_detector_version": "target-normal-v1",
                "d2_target_metrics": control["d2_target_metrics"],
                "matched_case_sample_key": str(case["sample_key"]),
                "match_tier": tier,
            }
        )
    frozen.sort(key=lambda item: str(item["sample_key"]))
    tier_counts = Counter(str(item["match_tier_name"]) for item in pairs)
    manifest = {
        "format_version": 2,
        "stage": "D2c-frozen-matched-normal-controls",
        "status": "frozen",
        "generated_at": timestamp(),
        "scope": "three-modality inference on model-unseen target-normal matched controls",
        "interpretation_limit": (
            "Controls are different trees matched without replacement. Source species and road "
            "matching are prioritized, but exact same-species same-road matches are not always available."
        ),
        "classes": classes,
        "summary": {
            "sample_count": len(frozen),
            "target_normal_pool_count": len(normal_pool),
            "match_tier_counts": dict(sorted(tier_counts.items())),
            "class_counts": dict(sorted(Counter(str(item["scientific_name"]) for item in frozen).items())),
            "road_count": len({str(item["road_id"]) for item in frozen}),
            "source_species_count": len({str(item["source_scientific_name"]) for item in frozen}),
        },
        "records": frozen,
    }
    manifest_path = output_root / "manifest.json"
    pairs_path = output_root / "pairs.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        pairs_path,
        {
            "format_version": 1,
            "status": "frozen",
            "generated_at": manifest["generated_at"],
            "case_manifest": str(case_path),
            "control_manifest": str(manifest_path),
            "pair_count": len(pairs),
            "pairs": sorted(pairs, key=lambda item: int(item["pair_id"])),
        },
    )
    validation = {
        "status": "passed",
        "generated_at": manifest["generated_at"],
        "case_count": len(case_records),
        "control_count": len(frozen),
        "unique_control_count": len({item["sample_key"] for item in frozen}),
        "pair_count": len(pairs),
        "same_model_class_all_pairs": all(
            str(case_records[int(row)]["scientific_name"])
            == str(normal_pool[int(column)]["model_class_name"])
            for row, column in zip(row_indices, column_indices, strict=True)
        ),
        "all_controls_target_normal": all(target_normal(item["d2_target_metrics"]) for item in frozen),
        "manifest_sha256": sha256_file(manifest_path),
        "pairs_sha256": sha256_file(pairs_path),
        "target_metrics_sha256": sha256_file(metrics_path),
    }
    atomic_json(output_root / "validation.json", validation)
    print(
        json.dumps(
            {
                "status": "passed",
                "target_normal_pool_count": len(normal_pool),
                "control_count": len(frozen),
                "match_tier_counts": dict(sorted(tier_counts.items())),
                "manifest": str(manifest_path),
                "pairs": str(pairs_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
