"""Validate a generated D1 rework batch and write a Chinese stage report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_D1_ROOT = DERIVED_ROOT / "d1_four_class_image_quality"
DEFAULT_C1_ROOT = DERIVED_ROOT / "c1_full_shared_dataset"
EXPECTED_POINT_KEYS = {
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def latest_batch(d1_root: Path) -> Path:
    batches = sorted(
        path
        for path in (d1_root / "rework_batches").glob("*")
        if (path / "rework_summary.json").is_file()
    )
    if not batches:
        raise FileNotFoundError("No complete D1 rework batch was found")
    return batches[-1]


def safe_path(batch_root: Path, relative: object) -> Path:
    path = (batch_root / str(relative)).resolve()
    try:
        path.relative_to(batch_root)
    except ValueError as error:
        raise ValueError(f"Path escapes the D1 rework batch: {relative}") from error
    return path


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        return image.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--c1-root", type=Path, default=DEFAULT_C1_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d1_root = args.d1_root.resolve()
    c1_root = args.c1_root.resolve()
    batch_root = (args.batch_root or latest_batch(d1_root)).resolve()
    manifest_path = batch_root / "replacement_review_manifest.json"
    review_path = batch_root / "replacement_user_review.json"
    inventory_path = batch_root / "candidate_inventory.csv"
    selection_path = batch_root / "replacement_selection.csv"
    summary_path = batch_root / "rework_summary.json"
    for path in (manifest_path, review_path, inventory_path, selection_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    errors: list[str] = []
    warnings: list[str] = []
    manifest = read_json(manifest_path)
    review = read_json(review_path)
    summary = read_json(summary_path)
    inventory = read_csv(inventory_path)
    selections = read_csv(selection_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Invalid replacement review manifest")
    records = [dict(record) for record in records]
    expected_total = int(summary.get("replacement_count", 0))
    expected_candidates = int(summary.get("candidate_count", 0))
    expected_actions = {
        str(key): int(value)
        for key, value in dict(summary.get("counts", {})).items()
    }
    if expected_total <= 0 or expected_candidates <= 0 or not expected_actions:
        errors.append("rework summary is missing expected batch counts")

    keys = [str(record.get("sample_key", "")) for record in records]
    if len(records) != expected_total:
        errors.append(
            f"replacement record count is {len(records)}, expected {expected_total}"
        )
    if not all(keys) or len(keys) != len(set(keys)):
        errors.append("replacement sample keys are missing or duplicated")
    if len(inventory) != expected_candidates:
        errors.append(
            f"candidate inventory count is {len(inventory)}, expected {expected_candidates}"
        )
    if len(selections) != expected_total:
        errors.append(
            f"selection row count is {len(selections)}, expected {expected_total}"
        )

    expected_manifest_hash = sha256_file(manifest_path)
    if review.get("reviewer") != "user":
        errors.append("replacement review is not user-owned")
    if review.get("source_manifest_sha256") != expected_manifest_hash:
        errors.append("replacement review is not bound to the replacement manifest")
    reviews = review.get("reviews")
    if not isinstance(reviews, dict):
        errors.append("replacement reviews object is invalid")

    source_review_path = d1_root / "user_visual_review.json"
    source_review_hash = sha256_file(source_review_path)
    if source_review_hash != manifest.get("source_d1_review_sha256"):
        errors.append("authoritative D1 review hash changed after rework generation")
    if source_review_hash != summary.get("source_review_sha256"):
        errors.append("rework summary has a different authoritative review hash")

    selected_inventory: dict[str, list[dict[str, str]]] = {}
    for row in inventory:
        if row.get("selected", "").casefold() == "true":
            selected_inventory.setdefault(row["sample_key"], []).append(row)
        for field in ("image_path", "overlay_path"):
            path = safe_path(batch_root, row[field])
            if not path.is_file():
                errors.append(f"missing candidate {field}: {row['sample_key']} {row[field]}")
            elif image_size(path) != (768, 512):
                errors.append(f"unexpected candidate dimensions: {row['sample_key']} {row[field]}")
        image_path = safe_path(batch_root, row["image_path"])
        if image_path.is_file() and sha256_file(image_path) != row["image_sha256"]:
            errors.append(f"candidate image hash mismatch: {row['sample_key']} {row['candidate_id']}")

    action_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    risk_count = 0
    risk_keys: list[str] = []
    for record in records:
        key = str(record["sample_key"])
        action = str(record["replacement_source_action"])
        action_counts[action] += 1
        class_counts[str(record["model_class_name"])] += 1
        split_counts[str(record["model_split"])] += 1
        method_counts[str(record["replacement_method"])] += 1
        if str(record["automatic_quality_tier"]) == "risk":
            risk_count += 1
            risk_keys.append(key)

        selected_rows = selected_inventory.get(key, [])
        if len(selected_rows) != 1:
            errors.append(f"{key} has {len(selected_rows)} selected candidate rows")
        elif selected_rows[0]["candidate_id"] != str(record["replacement_candidate_id"]):
            errors.append(f"selected candidate mismatch: {key}")

        image_path = safe_path(batch_root, record["image_path"])
        point_path = safe_path(batch_root, record["point_path"])
        comparison_path = safe_path(batch_root, record["comparison_image_path"])
        for path, label in (
            (image_path, "replacement image"),
            (point_path, "point asset"),
            (comparison_path, "comparison image"),
        ):
            if not path.is_file():
                errors.append(f"missing {label}: {key}")
        if image_path.is_file():
            if image_size(image_path) != (768, 512):
                errors.append(f"replacement dimensions are not 768x512: {key}")
            if sha256_file(image_path) != str(record["image_sha256"]):
                errors.append(f"replacement image hash mismatch: {key}")
        if comparison_path.is_file():
            image_size(comparison_path)
        if point_path.is_file():
            if sha256_file(point_path) != str(record["point_sha256"]):
                errors.append(f"replacement point hash mismatch: {key}")
            with np.load(point_path, allow_pickle=False) as archive:
                if set(archive.files) != EXPECTED_POINT_KEYS:
                    errors.append(f"point archive contract mismatch: {key}")
                if archive["points_xyz"].shape != (8192, 3):
                    errors.append(f"point tensor shape mismatch: {key}")

        original_image = c1_root / str(record["original_image_path"])
        original_point = c1_root / str(record.get("original_point_path", record["point_path"]))
        if not original_image.is_file():
            errors.append(f"missing original C1 image: {key}")
        elif sha256_file(original_image) != str(record["original_image_sha256"]):
            errors.append(f"original C1 image hash mismatch: {key}")
        source_point = c1_root / "assets" / "points" / f"{key}.npz"
        if not source_point.is_file() or (
            point_path.is_file() and sha256_file(source_point) != sha256_file(point_path)
        ):
            errors.append(f"copied point asset differs from C1 source: {key}")

        if action == "retry_alternate_view":
            if record["image_name"] == record["original_image_name"]:
                errors.append(f"alternate-view replacement reused the original panorama: {key}")
        elif action == "manual_recrop":
            if record["image_name"] != record["original_image_name"]:
                errors.append(f"manual recrop unexpectedly changed panorama: {key}")
        else:
            errors.append(f"unsupported replacement source action: {key} {action}")

    if dict(action_counts) != expected_actions:
        errors.append(f"action counts mismatch: {dict(action_counts)}")
    unreviewed_risk = [key for key in risk_keys if key not in reviews]
    accepted_risk = [
        key
        for key in risk_keys
        if key in reviews and str(reviews[key].get("action", "")) == "keep"
    ]
    if unreviewed_risk:
        warnings.append(
            f"{len(unreviewed_risk)} automatic-risk replacements still require user review"
        )
    if accepted_risk:
        warnings.append(
            f"{len(accepted_risk)} automatic-risk replacements were accepted by the user"
        )
    if reviews and isinstance(reviews, dict) and len(reviews) < len(records):
        warnings.append(
            f"replacement re-review has already started ({len(reviews)}/{len(records)})"
        )

    status = "failed" if errors else "passed_with_warnings" if warnings else "passed"
    result = {
        "stage": "D1-four-class-image-rework-validation",
        "status": status,
        "validated_at": timestamp(),
        "batch_root": str(batch_root),
        "replacement_count": len(records),
        "candidate_count": len(inventory),
        "action_counts": dict(action_counts),
        "method_counts": dict(method_counts),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "automatic_risk_count": risk_count,
        "replacement_reviewed": len(reviews) if isinstance(reviews, dict) else 0,
        "source_review_sha256": source_review_hash,
        "source_review_unchanged": not any("authoritative D1 review hash" in item for item in errors),
        "manifest_sha256": expected_manifest_hash,
        "errors": errors,
        "warnings": warnings,
    }
    (batch_root / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# D1 四分类图像返工候选阶段报告",
        "",
        f"- 验证时间：{result['validated_at']}",
        f"- 状态：**{status}**",
        f"- 返工样本：{len(records)}（换视角 {action_counts['retry_alternate_view']}，重裁剪 {action_counts['manual_recrop']}）",
        f"- 候选图像：{len(inventory)}",
        f"- 自动风险首选：{risk_count}",
        f"- 二次人工复核：{result['replacement_reviewed']}/{len(records)}",
        f"- 类别分布：{json.dumps(dict(class_counts), ensure_ascii=False)}",
        f"- 划分分布：{json.dumps(dict(split_counts), ensure_ascii=False)}",
        f"- 原 D1 用户审核哈希未改变：{result['source_review_unchanged']}",
        "",
        "## 验证结论",
        "",
        f"已检查 {len(records)} 个首选替代图、{len(inventory)} 个候选及叠加图、{len(records)} 个点云副本、对照图、清单哈希绑定和二次复核入口。原始 C1/WHU 数据及原 D1 用户审核记录未被覆盖。",
    ]
    if warnings:
        report_lines.extend(["", "## 警告", "", *[f"- {item}" for item in warnings]])
    if errors:
        report_lines.extend(["", "## 错误", "", *[f"- {item}" for item in errors]])
    report_lines.extend(
        [
            "",
            "科研灵感：多视角候选与人工返工结果可形成图像质量门控的监督信号，用于检验质量感知选择是否降低多模态负迁移。",
            "",
        ]
    )
    (batch_root / "D1返工候选阶段报告.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
