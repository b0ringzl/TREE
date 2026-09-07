"""Build the reviewed D1 four-class training/evaluation package without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_D1_ROOT = DERIVED_ROOT / "d1_four_class_image_quality"
DEFAULT_C1_ROOT = DERIVED_ROOT / "c1_full_shared_dataset"
DEFAULT_OUTPUT_ROOT = DERIVED_ROOT / "d1_four_class_training_package"
DEFAULT_EVALUATION = (
    DEFAULT_D1_ROOT
    / "evaluation_resolution"
    / "20260817_114510"
    / "resolved_multimodal_manifest.json"
)
DEFAULT_TRAINING_BATCH = (
    DEFAULT_D1_ROOT / "training_rework_batches" / "20260817_162429"
)
DEFAULT_REGISTRY = DEFAULT_D1_ROOT / "tracked_samples" / "tri_modal_probe_registry.json"
POINT_KEYS = (
    "points_xyz",
    "class_index",
    "benchmark_label_id",
    "raw_label_id",
    "centroid_xyz",
    "scale",
    "source_point_count",
    "sampled_unique_point_count",
    "sample_seed",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def ensure_unique(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    mapped = {str(record["sample_key"]): record for record in records}
    if len(mapped) != len(records):
        raise ValueError(f"Duplicate sample_key in {label}")
    return mapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--c1-root", type=Path, default=DEFAULT_C1_ROOT)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--training-batch", type=Path, default=DEFAULT_TRAINING_BATCH)
    parser.add_argument("--probe-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--package-id")
    return parser.parse_args()


def class_histogram(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record["scientific_name"]) for record in records)
    return dict(sorted(counts.items()))


def manifest_record(
    record: dict[str, Any],
    *,
    split: str,
    selection_tier: str,
    image_source: Path,
    image_sha256: str,
    point_source: Path,
    point_source_sha256: str,
    image_asset_path: str,
    point_asset_path: str,
    point_asset_sha256: str,
    resolution_reason: str,
) -> dict[str, Any]:
    class_index = int(record.get("model_class_index", record["class_index"]))
    scientific_name = str(
        record.get("model_class_name", record["scientific_name"])
    )
    return {
        "sample_key": str(record["sample_key"]),
        "split": split,
        "class_index": class_index,
        "scientific_name": scientific_name,
        "image_path": image_asset_path,
        "point_path": point_asset_path,
        "selection_tier": selection_tier,
        "resolution_reason": resolution_reason,
        "source_image_path": relative_to_project(image_source),
        "source_image_sha256": image_sha256,
        "packaged_image_sha256": image_sha256,
        "source_point_path": relative_to_project(point_source),
        "source_point_sha256": point_source_sha256,
        "packaged_point_sha256": point_asset_sha256,
        "benchmark_label_id": int(record["benchmark_label_id"]),
        "raw_label_id": int(record["raw_label_id"]),
        "road_id": str(record["road_id"]),
        "trajectory_id": str(record["trajectory_id"]),
        "tree_id": int(record["tree_id"]),
        "automatic_quality_tier": str(record.get("automatic_quality_tier", "")),
        "automatic_risk_score": float(record.get("automatic_risk_score", 0.0)),
        "automatic_risk_reasons": list(record.get("automatic_risk_reasons", [])),
    }


def view_record(record: dict[str, Any], prefix: str) -> dict[str, Any]:
    output = dict(record)
    output["image_path"] = f"{prefix}{record['image_path']}"
    output["point_path"] = f"{prefix}{record['point_path']}"
    return output


def build_manifest(
    records: list[dict[str, Any]],
    classes: list[dict[str, Any]],
    *,
    package_id: str,
    policy: str,
    path_prefix: str = "",
) -> dict[str, Any]:
    visible = [view_record(record, path_prefix) for record in records]
    split_sizes = Counter(str(record["split"]) for record in visible)
    return {
        "format_version": 1,
        "stage": "D1-four-class-reviewed-package",
        "status": "built_pending_validation",
        "generated_at": now_iso(),
        "package_id": package_id,
        "training_policy": policy,
        "image_size": [768, 512],
        "classes": classes,
        "summary": {
            "split_sizes": dict(sorted(split_sizes.items())),
            "class_counts": class_histogram(visible),
        },
        "records": visible,
    }


def main() -> None:
    args = parse_args()
    d1_root = args.d1_root.resolve()
    c1_root = args.c1_root.resolve()
    evaluation_path = args.evaluation_manifest.resolve()
    training_batch = args.training_batch.resolve()
    registry_path = args.probe_registry.resolve()
    output_root = args.output_root.resolve()
    package_id = args.package_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    package_root = output_root / package_id
    package_root.mkdir(parents=True, exist_ok=False)
    (package_root / "assets" / "images").mkdir(parents=True)
    (package_root / "assets" / "points").mkdir(parents=True)

    candidate_path = d1_root / "candidate_manifest.json"
    user_review_path = d1_root / "user_visual_review.json"
    replacement_manifest_path = training_batch / "replacement_review_manifest.json"
    replacement_review_path = training_batch / "replacement_user_review.json"
    candidate_payload = read_json(candidate_path)
    user_review_payload = read_json(user_review_path)
    replacement_manifest_payload = read_json(replacement_manifest_path)
    replacement_review_payload = read_json(replacement_review_path)
    evaluation_payload = read_json(evaluation_path)
    registry_payload = read_json(registry_path)

    classes = sorted(
        [dict(item) for item in candidate_payload["classes"]],
        key=lambda item: int(item["class_index"]),
    )
    if [int(item["class_index"]) for item in classes] != [0, 1, 2, 3]:
        raise ValueError("D1 classes must be contiguous indices 0..3")
    candidate_records = [dict(item) for item in candidate_payload["records"]]
    candidate_by_key = ensure_unique(candidate_records, "candidate manifest")
    user_reviews = user_review_payload.get("reviews")
    if not isinstance(user_reviews, dict):
        raise ValueError("D1 user review must contain a review dictionary")
    replacement_records = [
        dict(item) for item in replacement_manifest_payload["records"]
    ]
    replacement_by_key = ensure_unique(replacement_records, "replacement manifest")
    replacement_reviews = replacement_review_payload.get("reviews")
    if not isinstance(replacement_reviews, dict):
        raise ValueError("Replacement review must contain a review dictionary")
    if set(replacement_by_key) != set(replacement_reviews):
        raise ValueError("Replacement review is not complete or has unexpected keys")

    training_records = sorted(
        (
            record
            for record in candidate_records
            if str(record["model_split"]) == "train"
        ),
        key=lambda record: str(record["sample_key"]),
    )
    reviewed_training = {
        str(record["sample_key"]): record
        for record in training_records
        if str(record.get("review_scope", "")) == "training_risk"
    }
    if len(training_records) != 7001 or len(reviewed_training) != 120:
        raise ValueError(
            f"Unexpected D1 training inventory: {len(training_records)} total, "
            f"{len(reviewed_training)} reviewed-risk"
        )
    missing_reviews = sorted(set(reviewed_training) - set(user_reviews))
    if missing_reviews:
        raise ValueError(f"Missing training-risk reviews: {missing_reviews[:5]}")

    cohorts = registry_payload.get("cohorts", [])
    cohort = next(
        (
            item
            for item in cohorts
            if item.get("cohort_id") == "EXPOSURE_NEGATIVE_TRANSFER_V1"
        ),
        None,
    )
    if cohort is None:
        raise ValueError("Exposure negative-transfer cohort is missing")
    holdout_keys = {str(key) for key in cohort["sample_keys"]}
    if len(holdout_keys) != 7:
        raise ValueError(f"Expected 7 exposure probes, found {len(holdout_keys)}")
    if not holdout_keys <= set(candidate_by_key):
        raise ValueError("Exposure holdout contains unknown sample keys")

    accepted_clean: list[dict[str, Any]] = []
    accepted_silver: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}

    for record in training_records:
        key = str(record["sample_key"])
        image_source = c1_root / str(record["image_path"])
        image_hash = str(record["image_sha256"])
        resolution_reason = "automatic_quality_passed"
        final_state = "clean"
        review_rating = ""
        review_action = ""
        review_note = ""
        if key in reviewed_training:
            review = user_reviews[key]
            review_rating = str(review.get("rating", ""))
            review_action = str(review.get("action", ""))
            review_note = str(review.get("note", ""))
            if review_action == "keep":
                resolution_reason = "user_kept_original_training_image"
                final_state = "clean"
            elif review_action in {"exclude_image", "exclude_sample"}:
                resolution_reason = f"user_{review_action}"
                final_state = "excluded"
            elif review_action in {"retry_alternate_view", "manual_recrop"}:
                if key not in replacement_by_key:
                    raise ValueError(f"Missing replacement for {key}")
                replacement = replacement_by_key[key]
                replacement_review = replacement_reviews[key]
                replacement_action = str(replacement_review.get("action", ""))
                review_rating = str(replacement_review.get("rating", ""))
                review_note = str(replacement_review.get("note", ""))
                if replacement_action == "keep":
                    image_source = training_batch / str(replacement["image_path"])
                    image_hash = str(replacement["image_sha256"])
                    resolution_reason = "user_kept_training_replacement"
                    final_state = "clean"
                elif replacement_action in {"exclude_image", "exclude_sample"}:
                    resolution_reason = f"user_{replacement_action}_after_rework"
                    final_state = "excluded"
                else:
                    raise ValueError(
                        f"Unsupported final replacement action for {key}: "
                        f"{replacement_action}"
                    )
                review_action = replacement_action
            else:
                raise ValueError(f"Unsupported training review action for {key}: {review_action}")
        elif str(record["automatic_quality_tier"]) == "passed":
            final_state = "clean"
        elif str(record["automatic_quality_tier"]) == "risk":
            resolution_reason = "unreviewed_automatic_risk_silver_only"
            final_state = "silver_only"
        else:
            raise ValueError(f"Unsupported automatic quality tier for {key}")

        decision = {
            "sample_key": key,
            "scientific_name": str(record["model_class_name"]),
            "class_index": int(record["model_class_index"]),
            "final_state": final_state,
            "resolution_reason": resolution_reason,
            "review_rating": review_rating,
            "review_action": review_action,
            "review_note": review_note,
            "selected_image_source": image_source,
            "selected_image_sha256": image_hash,
            "point_source": c1_root / str(record["point_path"]),
            "point_source_sha256": str(record["point_sha256"]),
            "record": record,
        }
        if key in holdout_keys:
            if final_state not in {"clean", "silver_only"}:
                raise ValueError(f"Exposure probe was excluded by quality review: {key}")
            decision["final_state"] = "exposure_holdout"
            decision["resolution_reason"] = "user_registered_exposure_probe_holdout"
            holdout.append(decision)
        elif final_state == "clean":
            accepted_clean.append(decision)
            accepted_silver.append(decision)
        elif final_state == "silver_only":
            accepted_silver.append(decision)
        else:
            excluded.append(decision)
        decisions[key] = decision

    if set(replacement_by_key) != {
        key
        for key, record in reviewed_training.items()
        if str(user_reviews[key].get("action", ""))
        in {"retry_alternate_view", "manual_recrop"}
    }:
        raise ValueError("Replacement batch does not exactly match training rework actions")

    evaluation_records = sorted(
        [dict(item) for item in evaluation_payload["records"]],
        key=lambda record: str(record["sample_key"]),
    )
    ensure_unique(evaluation_records, "resolved evaluation manifest")
    if len(evaluation_records) != 241:
        raise ValueError(f"Expected 241 resolved evaluation records, found {len(evaluation_records)}")
    if {str(record["model_split"]) for record in evaluation_records} != {"val", "test"}:
        raise ValueError("Resolved evaluation manifest must contain only val/test")
    if set(record["sample_key"] for record in evaluation_records) & set(decisions):
        raise ValueError("Training and formal evaluation sample keys overlap")

    asset_jobs: list[dict[str, Any]] = accepted_silver + holdout
    for record in evaluation_records:
        asset_jobs.append(
            {
                "sample_key": str(record["sample_key"]),
                "scientific_name": str(record["model_class_name"]),
                "class_index": int(record["model_class_index"]),
                "final_state": str(record["model_split"]),
                "resolution_reason": str(record["resolution_reason"]),
                "selected_image_source": PROJECT_ROOT / str(record["resolved_image_path"]),
                "selected_image_sha256": str(record["resolved_image_sha256"]),
                "point_source": PROJECT_ROOT / str(record["resolved_point_path"]),
                "point_source_sha256": str(record["point_sha256"]),
                "record": record,
            }
        )
    if len({str(item["sample_key"]) for item in asset_jobs}) != len(asset_jobs):
        raise ValueError("Package asset jobs contain duplicate sample keys")

    packaged_by_key: dict[str, dict[str, Any]] = {}
    inventory_rows: list[dict[str, object]] = []
    point_bytes_written = 0
    for index, item in enumerate(asset_jobs, start=1):
        key = str(item["sample_key"])
        source_record = item["record"]
        image_source = Path(item["selected_image_source"]).resolve()
        point_source = Path(item["point_source"]).resolve()
        if not image_source.is_file() or not point_source.is_file():
            raise FileNotFoundError(f"Missing selected asset for {key}")
        if sha256_file(image_source) != str(item["selected_image_sha256"]):
            raise ValueError(f"Selected image hash mismatch: {key}")
        image_target = package_root / "assets" / "images" / f"{key}.jpg"
        point_target = package_root / "assets" / "points" / f"{key}.npz"
        os.link(image_source, image_target)
        with np.load(point_source, allow_pickle=False) as archive:
            if set(archive.files) != set(POINT_KEYS):
                raise ValueError(f"Unexpected point archive keys: {point_source}")
            values = {name: archive[name] for name in POINT_KEYS}
        points = values["points_xyz"]
        if points.shape != (8192, 3) or points.dtype != np.float32:
            raise ValueError(f"Unexpected point tensor for {key}: {points.shape} {points.dtype}")
        values["class_index"] = np.asarray(int(item["class_index"]), dtype=np.int64)
        point_temporary = Path(f"{point_target}.tmp")
        with point_temporary.open("wb") as stream:
            np.savez(stream, **values)
        point_temporary.replace(point_target)
        point_target_sha256 = sha256_file(point_target)
        point_bytes_written += point_target.stat().st_size
        split = str(item["final_state"])
        if split in {"clean", "silver_only"}:
            split = "train"
        selection_tier = str(item["final_state"])
        packaged = manifest_record(
            source_record,
            split=split,
            selection_tier=selection_tier,
            image_source=image_source,
            image_sha256=str(item["selected_image_sha256"]),
            point_source=point_source,
            point_source_sha256=str(item["point_source_sha256"]),
            image_asset_path=f"assets/images/{key}.jpg",
            point_asset_path=f"assets/points/{key}.npz",
            point_asset_sha256=point_target_sha256,
            resolution_reason=str(item["resolution_reason"]),
        )
        packaged_by_key[key] = packaged
        inventory_rows.append(
            {
                "sample_key": key,
                "class_index": int(item["class_index"]),
                "scientific_name": str(item["scientific_name"]),
                "selection_tier": selection_tier,
                "image_path": packaged["image_path"],
                "image_sha256": packaged["packaged_image_sha256"],
                "image_source_path": packaged["source_image_path"],
                "image_hardlink_count": image_target.stat().st_nlink,
                "point_path": packaged["point_path"],
                "point_sha256": packaged["packaged_point_sha256"],
                "point_source_path": packaged["source_point_path"],
                "point_source_sha256": packaged["source_point_sha256"],
            }
        )
        if index % 500 == 0 or index == len(asset_jobs):
            print(f"packaged assets {index}/{len(asset_jobs)}", flush=True)

    clean_records = [packaged_by_key[str(item["sample_key"])] for item in accepted_clean]
    silver_records = [packaged_by_key[str(item["sample_key"])] for item in accepted_silver]
    holdout_records = [packaged_by_key[str(item["sample_key"])] for item in holdout]
    formal_records = [packaged_by_key[str(item["sample_key"])] for item in asset_jobs if str(item["final_state"]) in {"val", "test"}]
    clean_records = sorted(clean_records + formal_records, key=lambda item: (str(item["split"]), str(item["sample_key"])))
    silver_records = sorted(silver_records + formal_records, key=lambda item: (str(item["split"]), str(item["sample_key"])))
    holdout_records = sorted(holdout_records, key=lambda item: str(item["sample_key"]))

    write_json(package_root / "classes.json", classes)
    write_json(
        package_root / "manifest.json",
        build_manifest(
            clean_records,
            classes,
            package_id=package_id,
            policy="clean_primary",
        ),
    )
    write_json(package_root / "silver" / "classes.json", classes)
    write_json(
        package_root / "silver" / "manifest.json",
        build_manifest(
            silver_records,
            classes,
            package_id=package_id,
            policy="expanded_silver_ablation",
            path_prefix="../",
        ),
    )
    write_json(package_root / "holdout" / "classes.json", classes)
    write_json(
        package_root / "holdout" / "manifest.json",
        build_manifest(
            holdout_records,
            classes,
            package_id=package_id,
            policy="exposure_negative_transfer_probe",
            path_prefix="../",
        ),
    )

    excluded_rows = [
        {
            "sample_key": item["sample_key"],
            "class_index": item["class_index"],
            "scientific_name": item["scientific_name"],
            "resolution_reason": item["resolution_reason"],
            "review_rating": item.get("review_rating", ""),
            "review_action": item.get("review_action", ""),
            "review_note": item.get("review_note", ""),
        }
        for item in excluded
    ]
    silver_only_rows = [
        {
            "sample_key": item["sample_key"],
            "class_index": item["class_index"],
            "scientific_name": item["scientific_name"],
            "automatic_risk_score": item["record"]["automatic_risk_score"],
            "automatic_risk_reasons": ";".join(item["record"]["automatic_risk_reasons"]),
        }
        for item in accepted_silver
        if str(item["final_state"]) == "silver_only"
    ]
    test_silver_rows = [
        {
            "sample_key": record["sample_key"],
            "class_index": record["model_class_index"],
            "scientific_name": record["model_class_name"],
            "model_split": record["model_split"],
            "automatic_quality_tier": record["automatic_quality_tier"],
            "image_path": relative_to_project(c1_root / str(record["image_path"])),
            "point_path": relative_to_project(c1_root / str(record["point_path"])),
        }
        for record in candidate_records
        if str(record["model_split"]) == "test_silver"
    ]
    write_csv(
        package_root / "asset_inventory.csv",
        inventory_rows,
        [
            "sample_key", "class_index", "scientific_name", "selection_tier",
            "image_path", "image_sha256", "image_source_path", "image_hardlink_count",
            "point_path", "point_sha256", "point_source_path", "point_source_sha256",
        ],
    )
    write_csv(
        package_root / "excluded_training_images.csv",
        excluded_rows,
        [
            "sample_key", "class_index", "scientific_name", "resolution_reason",
            "review_rating", "review_action", "review_note",
        ],
    )
    write_csv(
        package_root / "unreviewed_risk_silver_only.csv",
        silver_only_rows,
        [
            "sample_key", "class_index", "scientific_name",
            "automatic_risk_score", "automatic_risk_reasons",
        ],
    )
    write_csv(
        package_root / "test_silver_inventory.csv",
        test_silver_rows,
        [
            "sample_key", "class_index", "scientific_name", "model_split",
            "automatic_quality_tier", "image_path", "point_path",
        ],
    )

    review_metadata = []
    for record in clean_records:
        key = str(record["sample_key"])
        review = user_reviews.get(key, {})
        review_metadata.append(
            {
                "sample_key": key,
                "quality": str(review.get("rating", record["selection_tier"])),
                "note": str(review.get("note", record["resolution_reason"])),
            }
        )
    write_json(
        package_root / "quality_reviews_for_training.json",
        {"reviews": review_metadata},
    )

    source_paths = {
        "candidate_manifest": candidate_path,
        "user_visual_review": user_review_path,
        "training_replacement_manifest": replacement_manifest_path,
        "training_replacement_review": replacement_review_path,
        "resolved_evaluation_manifest": evaluation_path,
        "probe_registry": registry_path,
    }
    source_bindings = {
        name: {"path": relative_to_project(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    write_json(package_root / "source_bindings.json", source_bindings)

    summary = {
        "stage": "D1-four-class-package-construction",
        "status": "built_pending_validation",
        "generated_at": now_iso(),
        "package_id": package_id,
        "package_root": relative_to_project(package_root),
        "counts": {
            "candidate_train_total": len(training_records),
            "train_clean": len(accepted_clean),
            "train_silver": len(accepted_silver),
            "train_silver_only": len(silver_only_rows),
            "train_human_excluded": len(excluded),
            "exposure_holdout": len(holdout),
            "formal_val": sum(record["split"] == "val" for record in formal_records),
            "formal_test": sum(record["split"] == "test" for record in formal_records),
            "formal_evaluation_total": len(formal_records),
            "test_silver_inventory_only": len(test_silver_rows),
            "packaged_unique_assets": len(asset_jobs),
        },
        "class_counts": {
            "train_clean": class_histogram([packaged_by_key[str(item["sample_key"])] for item in accepted_clean]),
            "train_silver": class_histogram([packaged_by_key[str(item["sample_key"])] for item in accepted_silver]),
            "exposure_holdout": class_histogram(holdout_records),
            "formal_val": class_histogram([record for record in formal_records if record["split"] == "val"]),
            "formal_test": class_histogram([record for record in formal_records if record["split"] == "test"]),
        },
        "policies": {
            "default_manifest": "clean_primary",
            "silver_manifest": f"ablation_only_contains_{len(silver_only_rows)}_unreviewed_risk_images",
            "formal_evaluation": "only_user_accepted_resolved_val_test_images",
            "test_silver": "inventory_only_not_formal_metrics",
            "exposure_probe": "excluded_from_both_training_policies",
            "training_started": False,
        },
        "disk_change": {
            "image_assets": "NTFS hard links; no duplicate image payload blocks",
            "point_assets": "derived four-class NPZ copies with remapped class_index",
            "derived_point_bytes": point_bytes_written,
        },
        "source_bindings": source_bindings,
    }
    write_json(package_root / "package_summary.json", summary)

    report = [
        "# D1 四分类正式数据包构建报告",
        "",
        f"- 构建时间：{summary['generated_at']}",
        f"- 数据包：`{summary['package_root']}`",
        "- 状态：**built_pending_validation**",
        "- 训练状态：**未启动**",
        "",
        "## 结果",
        "",
        f"- clean 主训练集：{len(accepted_clean)}",
        f"- silver 扩展训练集：{len(accepted_silver)}（其中未人工复核风险样本 {len(silver_only_rows)}）",
        f"- 人工排除训练影像：{len(excluded)}",
        f"- 曝光负迁移留出探针：{len(holdout)}",
        f"- 正式验证集：{summary['counts']['formal_val']}",
        f"- 正式测试集：{summary['counts']['formal_test']}",
        f"- test_silver 仅清单：{len(test_silver_rows)}，不进入正式指标",
        "",
        "## 使用边界",
        "",
        "根目录 `manifest.json` 是默认 clean 方案；`silver/manifest.json` 仅供风险扩展消融。",
        "7 棵过暗/过亮探针树不在任一训练方案中，后续必须分别记录图像、点云、融合预测。",
        "验证集只用于选模；测试集不能用于挑选 epoch、随机种子或超参数。",
        "",
        "## 磁盘变化",
        "",
        f"图像使用 NTFS 硬链接；四分类标签重映射点云新增 {point_bytes_written / (1024 ** 2):.1f} MiB。",
        "原始 C1、WHU 数据和用户审核 JSON 均未覆盖。",
        "",
        "科研灵感：将曝光异常样本固定留出并与 clean/silver 两种训练政策并列，可把“图像质量导致融合负迁移”转化为可复现的配对消融假设。",
        "",
    ]
    (package_root / "D1四分类正式数据包构建报告.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
