"""Merge original and replacement D1 reviews into a final evaluation resolution.

This is a non-destructive manifest operation.  It never overwrites either
user-owned review checkpoint and never moves or deletes image/point assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_C1_ROOT = DERIVED_ROOT / "c1_full_shared_dataset"
DEFAULT_D1_ROOT = DERIVED_ROOT / "d1_four_class_image_quality"
EXPECTED_CLASSES = {
    "Cinnamomum camphora",
    "Lagerstroemia indica",
    "Magnolia grandiflora",
    "Other",
}
VALID_RATINGS = {"pass", "borderline", "reject"}
VALID_ACTIONS = {
    "keep",
    "exclude_image",
    "retry_alternate_view",
    "manual_recrop",
    "exclude_sample",
}
REWORK_ACTIONS = {"retry_alternate_view", "manual_recrop"}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def batch_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


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


def atomic_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def latest_rework_batch(d1_root: Path) -> Path:
    candidates = sorted(
        path
        for path in (d1_root / "rework_batches").glob("*")
        if (path / "replacement_review_manifest.json").is_file()
        and (path / "replacement_user_review.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError("No D1 replacement review batch was found")
    return candidates[-1]


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Resolved asset is outside the TREE project: {path}") from error


def validate_review(
    state: dict[str, object],
    manifest_path: Path,
    expected_keys: set[str],
    label: str,
) -> dict[str, dict[str, object]]:
    if state.get("reviewer") != "user":
        raise ValueError(f"{label} is not user-owned")
    if state.get("source_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"{label} is bound to a different manifest")
    reviews = state.get("reviews")
    if not isinstance(reviews, dict):
        raise ValueError(f"{label} has an invalid reviews object")
    actual_keys = set(str(key) for key in reviews)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{label} key mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    output: dict[str, dict[str, object]] = {}
    for key, value in reviews.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid review record: {key}")
        rating = str(value.get("rating", ""))
        action = str(value.get("action", ""))
        if rating not in VALID_RATINGS or action not in VALID_ACTIONS:
            raise ValueError(f"Invalid rating/action for {key}: {rating}/{action}")
        if rating == "pass" and action != "keep":
            raise ValueError(f"Passing sample does not use keep: {key}")
        if rating == "reject" and action == "keep":
            raise ValueError(f"Rejected sample uses keep: {key}")
        output[str(key)] = dict(value)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--c1-root", type=Path, default=DEFAULT_C1_ROOT)
    parser.add_argument("--rework-batch", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d1_root = args.d1_root.resolve()
    c1_root = args.c1_root.resolve()
    rework_batch = (args.rework_batch or latest_rework_batch(d1_root)).resolve()
    original_manifest_path = d1_root / "review_manifest.json"
    original_review_path = d1_root / "user_visual_review.json"
    replacement_manifest_path = rework_batch / "replacement_review_manifest.json"
    replacement_review_path = rework_batch / "replacement_user_review.json"
    for path in (
        original_manifest_path,
        original_review_path,
        replacement_manifest_path,
        replacement_review_path,
        c1_root / "shared_manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    original_manifest = read_json(original_manifest_path)
    original_records_raw = original_manifest.get("records")
    if not isinstance(original_records_raw, list):
        raise ValueError("Invalid D1 review manifest")
    evaluation_records = [
        dict(record)
        for record in original_records_raw
        if str(record.get("review_scope", "")) == "evaluation"
    ]
    if len(evaluation_records) != 320:
        raise ValueError(f"Expected 320 D1 evaluation records, got {len(evaluation_records)}")
    evaluation_by_key = {str(record["sample_key"]): record for record in evaluation_records}
    if len(evaluation_by_key) != len(evaluation_records):
        raise ValueError("D1 evaluation sample keys are duplicated")
    classes = {str(record["model_class_name"]) for record in evaluation_records}
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected D1 evaluation classes: {sorted(classes)}")

    original_state = read_json(original_review_path)
    all_original_reviews = original_state.get("reviews")
    if not isinstance(all_original_reviews, dict):
        raise ValueError("Invalid original D1 reviews object")
    evaluation_review_subset = {
        key: all_original_reviews[key]
        for key in evaluation_by_key
        if key in all_original_reviews
    }
    original_state_for_validation = {**original_state, "reviews": evaluation_review_subset}
    original_reviews = validate_review(
        original_state_for_validation,
        original_manifest_path,
        set(evaluation_by_key),
        "original D1 evaluation review",
    )

    replacement_manifest = read_json(replacement_manifest_path)
    replacement_records_raw = replacement_manifest.get("records")
    if not isinstance(replacement_records_raw, list):
        raise ValueError("Invalid replacement review manifest")
    replacement_records = [dict(record) for record in replacement_records_raw]
    replacement_by_key = {
        str(record["sample_key"]): record for record in replacement_records
    }
    if len(replacement_by_key) != 65:
        raise ValueError(f"Expected 65 replacement records, got {len(replacement_by_key)}")
    expected_rework_keys = {
        key
        for key, review in original_reviews.items()
        if str(review["action"]) in REWORK_ACTIONS
    }
    if set(replacement_by_key) != expected_rework_keys:
        raise ValueError("Replacement manifest does not match the original rework actions")
    replacement_state = read_json(replacement_review_path)
    replacement_reviews = validate_review(
        replacement_state,
        replacement_manifest_path,
        expected_rework_keys,
        "D1 replacement review",
    )

    source_hashes = {
        "original_review": sha256_file(original_review_path),
        "replacement_review": sha256_file(replacement_review_path),
        "original_manifest": sha256_file(original_manifest_path),
        "replacement_manifest": sha256_file(replacement_manifest_path),
        "c1_manifest": sha256_file(c1_root / "shared_manifest.json"),
    }
    check_payload = {
        "status": "preflight_passed",
        "evaluation_count": len(evaluation_records),
        "replacement_count": len(replacement_records),
        "replacement_reviewed": len(replacement_reviews),
        "source_hashes": source_hashes,
    }
    if args.check_only:
        print(json.dumps(check_payload, ensure_ascii=False, indent=2))
        return

    output_root = (
        args.output_root or d1_root / "evaluation_resolution" / batch_stamp()
    ).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse an evaluation resolution: {output_root}")
    output_root.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    accepted_records: list[dict[str, object]] = []
    for key in sorted(evaluation_by_key):
        original_record = evaluation_by_key[key]
        original_review = original_reviews[key]
        original_action = str(original_review["action"])
        selected_source = ""
        selected_image_path = ""
        selected_image_hash = ""
        replacement_review: dict[str, object] | None = None
        status = ""
        resolution_reason = ""

        if original_action == "keep":
            status = "multimodal_accepted"
            resolution_reason = "original_image_kept"
            selected_source = "original_c1"
            image_path = c1_root / str(original_record["image_path"])
            selected_image_path = project_relative(image_path)
            selected_image_hash = str(original_record["image_sha256"])
        elif original_action in {"exclude_image", "exclude_sample"}:
            status = "image_quarantined"
            resolution_reason = f"original_{original_action}"
        elif original_action in REWORK_ACTIONS:
            replacement_record = replacement_by_key[key]
            replacement_review = replacement_reviews[key]
            replacement_action = str(replacement_review["action"])
            if replacement_action == "keep":
                status = "multimodal_accepted"
                resolution_reason = "replacement_accepted"
                selected_source = "replacement_batch"
                image_path = rework_batch / str(replacement_record["image_path"])
                selected_image_path = project_relative(image_path)
                selected_image_hash = str(replacement_record["image_sha256"])
            elif replacement_action in {"exclude_image", "exclude_sample"}:
                status = "image_quarantined"
                resolution_reason = f"replacement_{replacement_action}"
            else:
                raise ValueError(f"Replacement still requires rework: {key}")
        else:
            raise ValueError(f"Unresolved original D1 action: {key} {original_action}")

        point_path = c1_root / str(original_record["point_path"])
        if not point_path.is_file() or sha256_file(point_path) != str(original_record["point_sha256"]):
            raise ValueError(f"C1 point asset mismatch: {key}")
        if status == "multimodal_accepted":
            resolved_image = PROJECT_ROOT / Path(selected_image_path)
            if not resolved_image.is_file() or sha256_file(resolved_image) != selected_image_hash:
                raise ValueError(f"Resolved image asset mismatch: {key}")

        row = {
            "sample_key": key,
            "model_class_index": original_record["model_class_index"],
            "model_class_name": original_record["model_class_name"],
            "model_split": original_record["model_split"],
            "road_id": original_record["road_id"],
            "trajectory_id": original_record["trajectory_id"],
            "tree_id": original_record["tree_id"],
            "resolution_status": status,
            "resolution_reason": resolution_reason,
            "selected_asset_source": selected_source,
            "selected_image_path": selected_image_path,
            "selected_image_sha256": selected_image_hash,
            "point_path": project_relative(point_path),
            "point_sha256": original_record["point_sha256"],
            "original_rating": original_review["rating"],
            "original_action": original_action,
            "original_reasons": "|".join(original_review.get("reasons", [])),
            "original_note": original_review.get("note", ""),
            "replacement_rating": replacement_review.get("rating", "") if replacement_review else "",
            "replacement_action": replacement_review.get("action", "") if replacement_review else "",
            "replacement_reasons": "|".join(replacement_review.get("reasons", [])) if replacement_review else "",
            "replacement_note": replacement_review.get("note", "") if replacement_review else "",
        }
        rows.append(row)
        if status == "multimodal_accepted":
            accepted_records.append(
                {
                    **original_record,
                    "resolved_asset_source": selected_source,
                    "resolved_image_path": selected_image_path,
                    "resolved_image_sha256": selected_image_hash,
                    "resolved_point_path": project_relative(point_path),
                    "resolution_reason": resolution_reason,
                }
            )

    status_counts = Counter(str(row["resolution_status"]) for row in rows)
    reason_counts = Counter(str(row["resolution_reason"]) for row in rows)
    class_status: dict[str, dict[str, int]] = {}
    for class_name in sorted(EXPECTED_CLASSES):
        class_rows = [row for row in rows if row["model_class_name"] == class_name]
        class_status[class_name] = {
            "total": len(class_rows),
            "multimodal_accepted": sum(
                row["resolution_status"] == "multimodal_accepted" for row in class_rows
            ),
            "image_quarantined": sum(
                row["resolution_status"] == "image_quarantined" for row in class_rows
            ),
        }
    split_status: dict[str, dict[str, int]] = {}
    for split in ("val", "test"):
        split_rows = [row for row in rows if row["model_split"] == split]
        split_status[split] = {
            "total": len(split_rows),
            "multimodal_accepted": sum(
                row["resolution_status"] == "multimodal_accepted" for row in split_rows
            ),
            "image_quarantined": sum(
                row["resolution_status"] == "image_quarantined" for row in split_rows
            ),
        }

    atomic_csv(output_root / "evaluation_resolution.csv", rows)
    atomic_csv(
        output_root / "multimodal_evaluation.csv",
        [row for row in rows if row["resolution_status"] == "multimodal_accepted"],
    )
    atomic_csv(
        output_root / "image_quarantine.csv",
        [row for row in rows if row["resolution_status"] == "image_quarantined"],
    )
    atomic_csv(output_root / "point_evaluation.csv", rows)
    resolved_manifest_path = output_root / "resolved_multimodal_manifest.json"
    atomic_json(
        resolved_manifest_path,
        {
            "format_version": 1,
            "stage": "D1-four-class-resolved-multimodal-evaluation",
            "status": "complete",
            "generated_at": timestamp(),
            "project_root": str(PROJECT_ROOT),
            "source_hashes": source_hashes,
            "classes": original_manifest.get("classes", []),
            "records": accepted_records,
        },
    )

    if sha256_file(original_review_path) != source_hashes["original_review"]:
        raise RuntimeError("Original D1 review changed during resolution")
    if sha256_file(replacement_review_path) != source_hashes["replacement_review"]:
        raise RuntimeError("Replacement D1 review changed during resolution")

    summary = {
        "stage": "D1-four-class-evaluation-resolution",
        "status": "complete",
        "generated_at": timestamp(),
        "output_root": str(output_root),
        "evaluation_total": len(rows),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "class_status": class_status,
        "split_status": split_status,
        "replacement_reviewed": len(replacement_reviews),
        "source_hashes": source_hashes,
        "source_reviews_unchanged": True,
        "resolved_manifest": str(resolved_manifest_path),
        "resolved_manifest_sha256": sha256_file(resolved_manifest_path),
        "next_gate": "complete the 120-sample D1 training-risk human review",
    }
    atomic_json(output_root / "evaluation_resolution_summary.json", summary)

    report = [
        "# D1 四分类评估图像最终解析报告",
        "",
        f"- 生成时间：{summary['generated_at']}",
        f"- 正式评估样本：{len(rows)}",
        f"- 多模态可用：{status_counts['multimodal_accepted']}",
        f"- 图像隔离：{status_counts['image_quarantined']}",
        f"- 原图直接保留：{reason_counts['original_image_kept']}",
        f"- 替代图复核通过：{reason_counts['replacement_accepted']}",
        f"- 替代图复核后排除：{reason_counts['replacement_exclude_image'] + reason_counts['replacement_exclude_sample']}",
        "",
        "## 类别覆盖",
        "",
        "| 类别 | 原始评估数 | 多模态可用 | 图像隔离 |",
        "|---|---:|---:|---:|",
    ]
    for class_name, counts in class_status.items():
        report.append(
            f"| {class_name} | {counts['total']} | {counts['multimodal_accepted']} | {counts['image_quarantined']} |"
        )
    report.extend(
        [
            "",
            "## 安全边界",
            "",
            "两份用户审核 JSON、C1 图像、C1 点云和 WHU 原始数据均未覆盖、移动或删除。本阶段只生成解析清单，未启动训练或重建缓存。",
            "",
            "## 下一关",
            "",
            "继续完成 120 张训练高风险样本的人工审核。评估解析结果稳定并不等于已经批准构建训练包或启动模型训练。",
            "",
            "科研灵感：类别级隔离比例的差异可作为质量门控偏差审计指标，检验质检策略是否对特定树种产生选择偏差。",
            "",
        ]
    )
    (output_root / "D1评估图像最终解析报告.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
