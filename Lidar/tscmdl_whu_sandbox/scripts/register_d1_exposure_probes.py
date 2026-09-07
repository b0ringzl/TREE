"""Register the user-selected D1 exposure probes for tri-modal comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_C1_ROOT = DERIVED_ROOT / "c1_full_shared_dataset"
DEFAULT_D1_ROOT = DERIVED_ROOT / "d1_four_class_image_quality"
COHORT_ID = "EXPOSURE_NEGATIVE_TRANSFER_V1"
PROBES = (
    ("TRI_MODAL_PROBE_001", "10_6_14017", "重点跟踪树-14017", "dark"),
    ("TRI_MODAL_PROBE_002", "10_6_14078", "过暗探针-14078", "dark"),
    ("TRI_MODAL_PROBE_003", "10_6_14080", "过暗探针-14080", "dark"),
    ("TRI_MODAL_PROBE_004", "23_1_2496", "过亮探针-2496", "bright"),
    ("TRI_MODAL_PROBE_005", "23_1_2501", "过亮探针-2501", "bright"),
    ("TRI_MODAL_PROBE_006", "23_1_2681", "过亮探针-2681", "bright"),
    ("TRI_MODAL_PROBE_007", "23_1_2687", "过亮探针-2687", "bright"),
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def result_template() -> dict[str, object]:
    return {
        "status": "pending",
        "checkpoint": None,
        "top1_prediction": None,
        "correct": None,
        "target_class_probability": None,
        "top1_confidence": None,
        "target_class_rank": None,
        "evaluation_context": {
            "evaluation_manifest": None,
            "evaluation_sample_count": None,
            "accuracy": None,
            "macro_f1": None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--c1-root", type=Path, default=DEFAULT_C1_ROOT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d1_root = args.d1_root.resolve()
    c1_root = args.c1_root.resolve()
    tracked_root = d1_root / "tracked_samples"
    registry_path = tracked_root / "tri_modal_probe_registry.json"
    manifest_path = d1_root / "review_manifest.json"
    review_path = d1_root / "user_visual_review.json"
    for path in (registry_path, manifest_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    old_registry = read_json(registry_path)
    manifest = read_json(manifest_path)
    review_state = read_json(review_path)
    records_raw = manifest.get("records")
    reviews_raw = review_state.get("reviews")
    if not isinstance(records_raw, list) or not isinstance(reviews_raw, dict):
        raise ValueError("Invalid D1 manifest or user review")
    record_by_key = {str(record["sample_key"]): dict(record) for record in records_raw}
    review_by_key = {str(key): dict(value) for key, value in reviews_raw.items()}
    old_samples = old_registry.get("samples", [])
    if not isinstance(old_samples, list):
        raise ValueError("Invalid existing probe registry")
    old_by_key = {str(sample["sample_key"]): dict(sample) for sample in old_samples}

    source_hashes = {
        "d1_review_manifest": sha256_file(manifest_path),
        "d1_user_visual_review": sha256_file(review_path),
        "c1_shared_manifest": sha256_file(c1_root / "shared_manifest.json"),
    }
    probe_samples: list[dict[str, object]] = []
    for probe_id, key, user_label, exposure_group in PROBES:
        if key not in record_by_key or key not in review_by_key:
            raise ValueError(f"Probe is missing its record or user review: {key}")
        record = record_by_key[key]
        review = review_by_key[key]
        if record.get("review_scope") != "training_risk" or record.get("model_split") != "train":
            raise ValueError(f"Probe is not a D1 training-risk train sample: {key}")
        if review.get("rating") != "pass" or review.get("action") != "keep":
            raise ValueError(f"Probe was not accepted by the user: {key}")
        risk_reasons = {str(value) for value in record.get("automatic_risk_reasons", [])}
        expected_reason = (
            "excessive_dark_pixels" if exposure_group == "dark" else "excessive_bright_pixels"
        )
        if expected_reason not in risk_reasons:
            raise ValueError(f"Probe exposure group does not match its automatic risks: {key}")
        image_path = c1_root / str(record["image_path"])
        point_path = c1_root / str(record["point_path"])
        if sha256_file(image_path) != str(record["image_sha256"]):
            raise ValueError(f"Probe image hash mismatch: {key}")
        if sha256_file(point_path) != str(record["point_sha256"]):
            raise ValueError(f"Probe point hash mismatch: {key}")

        previous = old_by_key.get(key, {})
        previous_results = previous.get("results")
        previous_role = previous.get("experimental_role", {})
        if not isinstance(previous_role, dict):
            previous_role = {}
        results = (
            previous_results
            if isinstance(previous_results, dict)
            else {
                "image_only": result_template(),
                "point_only": result_template(),
                "fusion": result_template(),
            }
        )
        metrics = record.get("automatic_metrics", {})
        probe_samples.append(
            {
                "probe_id": probe_id,
                "user_label": user_label,
                "registered_by": "user_request",
                "registered_at": previous.get("registered_at", timestamp()),
                "cohort_ids": [COHORT_ID],
                "sample_key": key,
                "road_id": str(record["road_id"]),
                "trajectory_id": str(record["trajectory_id"]),
                "tree_id": int(record["tree_id"]),
                "ground_truth": {
                    "model_class_index": int(record["model_class_index"]),
                    "model_class_name": str(record["model_class_name"]),
                    "source_benchmark_label_id": int(record["benchmark_label_id"]),
                    "source_scientific_name": str(record["source_scientific_name"]),
                },
                "data_scope": {
                    "model_split": "train",
                    "review_scope": "training_risk",
                    "source_point_count": int(record["source_point_count"]),
                    "sampled_point_count": int(record["sampled_unique_point_count"]),
                    "camera_distance_m": float(record["camera_distance_m"]),
                },
                "assets": {
                    "image_path": project_relative(image_path),
                    "image_sha256": str(record["image_sha256"]),
                    "point_path": project_relative(point_path),
                    "point_sha256": str(record["point_sha256"]),
                },
                "quality": {
                    "exposure_group": exposure_group,
                    "automatic_quality_tier": str(record["automatic_quality_tier"]),
                    "automatic_risk_score": float(record["automatic_risk_score"]),
                    "automatic_risk_reasons": sorted(risk_reasons),
                    "mean_luminance": float(metrics["mean_luminance"]),
                    "dark_ratio": float(metrics["dark_ratio"]),
                    "bright_ratio": float(metrics["bright_ratio"]),
                    "dynamic_range_90": float(metrics["dynamic_range_90"]),
                    "edge_energy": float(metrics["edge_energy"]),
                    "user_rating": str(review["rating"]),
                    "user_action": str(review["action"]),
                    "user_reviewed_at": str(review["reviewed_at"]),
                },
                "experimental_role": {
                    "fixed_case_probe": True,
                    "recommended_holdout": True,
                    "training_exclusion_applied": bool(
                        previous_role.get("training_exclusion_applied", False)
                    ),
                    "interpretation_until_holdout": previous_role.get(
                        "interpretation_until_holdout",
                        "training-sample diagnostic only",
                    ),
                    **(
                        {"holdout_package_id": previous_role["holdout_package_id"]}
                        if "holdout_package_id" in previous_role
                        else {}
                    ),
                },
                "results": results,
                "paired_comparison": {
                    "status": "pending",
                    "negative_transfer_event": None,
                    "fusion_minus_point_correct": None,
                    "fusion_minus_point_target_probability": None,
                    "image_minus_point_correct": None,
                },
            }
        )

    groups = Counter(sample["quality"]["exposure_group"] for sample in probe_samples)
    classes = Counter(sample["ground_truth"]["model_class_name"] for sample in probe_samples)
    old_policy = old_registry.get("reporting_policy", {})
    if not isinstance(old_policy, dict):
        old_policy = {}
    old_cohort = next(
        (
            item
            for item in old_registry.get("cohorts", [])
            if isinstance(item, dict) and item.get("cohort_id") == COHORT_ID
        ),
        {},
    )
    exclusion_applied = bool(old_policy.get("training_exclusion_applied", False))
    registry = {
        "schema_version": 2,
        "stage": "D1-tri-modal-probe-registry",
        "status": old_registry.get("status", "registered_pending_predictions"),
        "created_at": old_registry.get("created_at", timestamp()),
        "updated_at": timestamp(),
        "join_key": "sample_key",
        "required_prediction_branches": ["image_only", "point_only", "fusion"],
        "source_hashes": source_hashes,
        "reporting_policy": {
            "per_sample_fields": [
                "top1_prediction",
                "correct",
                "target_class_probability",
                "top1_confidence",
                "target_class_rank",
            ],
            "branch_evaluation_context_fields": [
                "evaluation_manifest",
                "evaluation_sample_count",
                "accuracy",
                "macro_f1",
            ],
            "accuracy_definition": "Accuracy is aggregate; one tree is correct/incorrect with confidence.",
            "checkpoint_selection": "All checkpoints must be selected by validation data only.",
            "comparison_requirement": "Use the same locked four-class label order and the same sample keys for all branches.",
            "negative_transfer_definition": "Primary cohort endpoint is fusion accuracy minus point-only accuracy. A per-sample negative-transfer event is point-only correct and fusion incorrect.",
            "causal_limit": "Exposure causality requires a held-out exposure cohort and matched normal-exposure controls; these seven fixed probes alone are case-study evidence.",
            "generalization_warning": old_policy.get(
                "generalization_warning",
                "All probes are currently train samples and remain training diagnostics until formally excluded before package freeze.",
            ),
            "training_exclusion_applied": exclusion_applied,
            **(
                {"holdout_binding": old_policy["holdout_binding"]}
                if "holdout_binding" in old_policy
                else {}
            ),
        },
        "cohorts": [
            {
                "cohort_id": COHORT_ID,
                "title": "过暗/过亮影像对融合相对点云表现的负迁移探针",
                "status": old_cohort.get("status", "registered_pending_predictions"),
                "hypothesis": "Exposure-degraded images can reduce fusion performance relative to the point-only branch.",
                "sample_keys": [sample["sample_key"] for sample in probe_samples],
                "sample_count": len(probe_samples),
                "exposure_group_counts": dict(groups),
                "class_counts": dict(classes),
                "primary_endpoint": "accuracy_fusion_minus_accuracy_point",
                "secondary_endpoints": [
                    "negative_transfer_event_rate",
                    "mean_target_probability_fusion_minus_point",
                    "accuracy_image_minus_point",
                ],
                "stratify_by": ["exposure_group", "ground_truth_class"],
                "normal_exposure_control_status": "pending_selection",
                "recommended_holdout": True,
                "training_exclusion_applied": bool(
                    old_cohort.get("training_exclusion_applied", exclusion_applied)
                ),
                **(
                    {"holdout_binding": old_cohort["holdout_binding"]}
                    if "holdout_binding" in old_cohort
                    else {}
                ),
                "conclusion_status": "not_tested",
            }
        ],
        "samples": probe_samples,
    }
    check = {
        "status": "preflight_passed",
        "cohort_id": COHORT_ID,
        "sample_count": len(probe_samples),
        "exposure_group_counts": dict(groups),
        "class_counts": dict(classes),
        "all_user_accepted": True,
        "source_hashes": source_hashes,
    }
    if args.check_only:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return

    atomic_json(registry_path, registry)
    csv_rows = []
    for sample in probe_samples:
        csv_rows.append(
            {
                "cohort_id": COHORT_ID,
                "probe_id": sample["probe_id"],
                "user_label": sample["user_label"],
                "sample_key": sample["sample_key"],
                "road_id": sample["road_id"],
                "trajectory_id": sample["trajectory_id"],
                "tree_id": sample["tree_id"],
                "ground_truth_class_index": sample["ground_truth"]["model_class_index"],
                "ground_truth_class_name": sample["ground_truth"]["model_class_name"],
                "exposure_group": sample["quality"]["exposure_group"],
                "mean_luminance": sample["quality"]["mean_luminance"],
                "dark_ratio": sample["quality"]["dark_ratio"],
                "bright_ratio": sample["quality"]["bright_ratio"],
                "automatic_risk_score": sample["quality"]["automatic_risk_score"],
                "automatic_risk_reasons": "|".join(sample["quality"]["automatic_risk_reasons"]),
                "model_split": sample["data_scope"]["model_split"],
                "user_rating": sample["quality"]["user_rating"],
                "user_action": sample["quality"]["user_action"],
                "image_path": sample["assets"]["image_path"],
                "image_sha256": sample["assets"]["image_sha256"],
                "point_path": sample["assets"]["point_path"],
                "point_sha256": sample["assets"]["point_sha256"],
                "image_only_status": sample["results"]["image_only"]["status"],
                "point_only_status": sample["results"]["point_only"]["status"],
                "fusion_status": sample["results"]["fusion"]["status"],
                "recommended_holdout": True,
                "training_exclusion_applied": sample["experimental_role"][
                    "training_exclusion_applied"
                ],
            }
        )
    atomic_csv(tracked_root / "tri_modal_probe_registry.csv", csv_rows)
    atomic_json(
        tracked_root / "exposure_negative_transfer_plan.json",
        {
            "schema_version": 1,
            "stage": "D1-exposure-negative-transfer-plan",
            "generated_at": timestamp(),
            "source_registry": str(registry_path),
            "source_registry_sha256": sha256_file(registry_path),
            "cohort": registry["cohorts"][0],
            "analysis_contract": {
                "required_branches": registry["required_prediction_branches"],
                "paired_join_key": "sample_key",
                "primary_endpoint": "accuracy_fusion_minus_accuracy_point",
                "negative_transfer_event": "point_correct == true and fusion_correct == false",
                "report_confidence": True,
                "report_target_class_probability": True,
                "report_by_exposure_group": True,
                "matched_normal_controls_required_for_causal_claim": True,
            },
        },
    )
    print(json.dumps({**check, "status": "registered"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
