#!/usr/bin/env python3
"""Evaluate VMMS detector and species prelabels against the first reviewed frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELER = PROJECT_ROOT / "derived" / "exhaustive_species_frame_labeler"
DEFAULT_SOURCE = PROJECT_ROOT / "derived" / "exhaustive_species_review_20260906"
DEFAULT_COORDINATES = PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv"
STREAM_IDS = (
    "hewentian_pano",
    "jianshazui_pano_1",
    "jianshazui_pano_2",
    "stubbs_pano_0",
    "stubbs_pano_1",
)
UNKNOWN = {"", "Unknown / 待定", "Unknown 未知"}
BANYAN_LABEL = "榕树"


def normalize_species(value: object) -> str:
    name = str(value or "").strip()
    folded = name.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return BANYAN_LABEL
    if any(token in name for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return BANYAN_LABEL
    return BANYAN_LABEL if name == "榕樹" else name


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def bbox(label: dict[str, Any]) -> tuple[float, float, float, float]:
    existing = label.get("bbox") or {}
    if all(key in existing for key in ("left", "top", "right", "bottom")):
        return tuple(float(existing[key]) for key in ("left", "top", "right", "bottom"))
    if all(key in existing for key in ("x_center", "y_center", "width", "height")):
        x_center, y_center = float(existing["x_center"]), float(existing["y_center"])
        width, height = float(existing["width"]), float(existing["height"])
        return (
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        )
    points = label.get("points") or []
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def box_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    left_a, top_a, right_a, bottom_a = bbox(first)
    left_b, top_b, right_b, bottom_b = bbox(second)
    intersection = max(0.0, min(right_a, right_b) - max(left_a, left_b)) * max(
        0.0, min(bottom_a, bottom_b) - max(top_a, top_b)
    )
    area_a = max(0.0, right_a - left_a) * max(0.0, bottom_a - top_a)
    area_b = max(0.0, right_b - left_b) * max(0.0, bottom_b - top_b)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_labels(
    source_labels: list[dict[str, Any]],
    final_labels: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[tuple[int, int, str, float]], set[int], set[int]]:
    matches: list[tuple[int, int, str, float]] = []
    used_source: set[int] = set()
    used_final: set[int] = set()
    final_by_id = {
        str(label.get("label_id")): index
        for index, label in enumerate(final_labels)
        if label.get("label_id")
    }
    for source_index, label in enumerate(source_labels):
        final_index = final_by_id.get(str(label.get("label_id")))
        if final_index is None or final_index in used_final:
            continue
        matches.append((source_index, final_index, "label_id", box_iou(label, final_labels[final_index])))
        used_source.add(source_index)
        used_final.add(final_index)

    candidates: list[tuple[float, int, int]] = []
    for source_index, source_label in enumerate(source_labels):
        if source_index in used_source:
            continue
        for final_index, final_label in enumerate(final_labels):
            if final_index in used_final:
                continue
            overlap = box_iou(source_label, final_label)
            if overlap >= threshold:
                candidates.append((overlap, source_index, final_index))
    for overlap, source_index, final_index in sorted(candidates, reverse=True):
        if source_index in used_source or final_index in used_final:
            continue
        matches.append((source_index, final_index, "bbox_iou", overlap))
        used_source.add(source_index)
        used_final.add(final_index)
    return matches, used_source, used_final


def sampled_frame_keys(path: Path, spacing_m: float) -> list[str]:
    grouped: dict[str, list[dict[str, str]]] = {stream_id: [] for stream_id in STREAM_IDS}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] in grouped:
                grouped[row["stream_id"]].append(row)

    sampled: list[str] = []
    for stream_id in STREAM_IDS:
        rows = sorted(grouped[stream_id], key=lambda row: int(row["seq_id"]))
        if not rows:
            continue
        route_distance = 0.0
        previous: tuple[float, float] | None = None
        last_sample_distance = 0.0
        stream_samples: list[str] = []
        for index, row in enumerate(rows):
            point = (float(row["hk80_easting"]), float(row["hk80_northing"]))
            if previous is not None:
                route_distance += math.dist(previous, point)
            previous = point
            if index == 0 or spacing_m <= 0 or route_distance - last_sample_distance >= spacing_m:
                stream_samples.append(f"{stream_id}__{row['frame_id']}")
                last_sample_distance = route_distance
        last_key = f"{stream_id}__{rows[-1]['frame_id']}"
        if last_key not in stream_samples:
            stream_samples.append(last_key)
        sampled.extend(stream_samples)
    return sampled


def source_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record_path in path.glob("records/*.json"):
        record = load_json(record_path)
        records[f"{record['stream_id']}__{record['frame_id']}"] = record
    return records


def prediction_confidence(row: dict[str, Any]) -> float | None:
    species = normalize_species(row.get("species"))
    for candidate_key, confidence_key in (
        ("domain_species", "domain_confidence"),
        ("web_species", "web_confidence"),
    ):
        if normalize_species(row.get(candidate_key)) != species:
            continue
        try:
            confidence = float(row.get(confidence_key) or "")
        except ValueError:
            continue
        if 0.0 <= confidence <= 1.0:
            return confidence
    return None


def load_predictions(source_dir: Path, records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    assistant_csv = source_dir / "assistant_prelabels_20260906" / "assistant_prelabels.csv"
    with assistant_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            predictions[row["label_id"]] = {
                "species": normalize_species(row["species"]),
                "tier": row["tier"],
                "method": row["method"],
                "confidence": prediction_confidence(row),
            }

    for record in records.values():
        for label in record.get("labels") or []:
            label_id = str(label.get("label_id") or "")
            species = normalize_species(label.get("species"))
            if not label_id or label_id in predictions or species in UNKNOWN:
                continue
            if label.get("species_confidence_tier") == "human_verified":
                continue
            hint = label.get("vmms_domain_classifier_suggestion") or {}
            confidence = None
            if normalize_species(hint.get("top1_species")) == species:
                try:
                    confidence = float(hint.get("confidence"))
                except (TypeError, ValueError):
                    confidence = None
            predictions[label_id] = {
                "species": species,
                "tier": str(label.get("species_confidence_tier") or "auto").replace("assistant_", ""),
                "method": label.get("species_method") or "existing_auto_prelabel",
                "confidence": confidence,
            }
    return predictions


def confidence_band(value: float | None) -> str:
    if value is None:
        return "无可比模型分数"
    if value >= 0.99:
        return "≥0.99"
    if value >= 0.90:
        return "0.90–0.99"
    if value >= 0.70:
        return "0.70–0.90"
    if value >= 0.50:
        return "0.50–0.70"
    return "<0.50"


def metric_row(counts: Counter[str]) -> dict[str, Any]:
    selected = counts["selected"]
    correct = counts["correct"]
    return {"selected": selected, "correct": correct, "accuracy": safe_div(correct, selected)}


def comparison_row(counts: Counter[str]) -> dict[str, Any]:
    before = counts["before"]
    retained = counts["retained"]
    correct = counts["correct"]
    return {
        "before": before,
        "retained": retained,
        "deleted": counts["deleted"],
        "species_corrected": counts["species_corrected"],
        "correct": correct,
        "accuracy_including_deleted": safe_div(correct, before),
        "retained_top1_accuracy": safe_div(correct, retained),
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler-root", type=Path, default=DEFAULT_LABELER)
    parser.add_argument("--source-review", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--reviewed-frame-count", type=int, default=280)
    parser.add_argument("--spacing-m", type=float, default=5.0)
    parser.add_argument("--iou-threshold", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    cfg = parser.parse_args()

    labeler = cfg.labeler_root.resolve()
    source_dir = cfg.source_review.resolve()
    output_dir = cfg.output_dir.resolve()
    review_state = load_json(labeler / "runtime" / "frame_review_state.json")
    records = source_records(source_dir)
    predictions = load_predictions(source_dir, records)
    all_sampled = sampled_frame_keys(cfg.coordinates.resolve(), cfg.spacing_m)
    primary_keys = all_sampled[: cfg.reviewed_frame_count]
    reviewed_keys = [key for key in primary_keys if key in review_state]
    missing_keys = [key for key in primary_keys if key not in review_state]
    formal_record_keys = {
        str(load_json(path).get("frame_key") or "")
        for path in (labeler / "annotations" / "records").glob("**/*.json")
    }
    saved_outside_primary = sorted(key for key in formal_record_keys if key not in set(primary_keys))

    status_counts: Counter[str] = Counter()
    instance = Counter()
    tier_counts: dict[str, Counter[str]] = defaultdict(Counter)
    band_counts: dict[str, Counter[str]] = defaultdict(Counter)
    species_counts: dict[str, Counter[str]] = defaultdict(Counter)
    confusions: Counter[tuple[str, str]] = Counter()
    instance_rows: list[dict[str, Any]] = []

    for review_index, frame_key in enumerate(primary_keys, start=1):
        final_record = review_state.get(frame_key)
        if not final_record:
            continue
        status = final_record.get("frame_status", "")
        status_counts[status] += 1
        if status == "unusable":
            continue
        source_record = records.get(frame_key, {})
        source_labels = source_record.get("labels") or []
        final_labels = final_record.get("labels") or []
        matches, _, _ = match_labels(source_labels, final_labels, cfg.iou_threshold)
        final_for_source = {source_index: final_index for source_index, final_index, _, _ in matches}

        auto_source_indices = [
            index
            for index, label in enumerate(source_labels)
            if label.get("species_confidence_tier") != "human_verified"
        ]
        for source_index in auto_source_indices:
            source_label = source_labels[source_index]
            prediction = predictions.get(str(source_label.get("label_id") or ""))
            if not prediction:
                instance["missing_pre_review_prediction"] += 1
                continue
            predicted_species = normalize_species(prediction["species"])
            tier = str(prediction.get("tier") or "unknown")
            band = confidence_band(prediction.get("confidence"))
            instance["predictions_before"] += 1
            for container, key in ((tier_counts, tier), (band_counts, band), (species_counts, predicted_species)):
                container[key]["before"] += 1

            final_index = final_for_source.get(source_index)
            if final_index is None:
                final_species = "已删除实例"
                correct = False
                outcome = "deleted"
                instance["predictions_deleted"] += 1
                for container, key in ((tier_counts, tier), (band_counts, band), (species_counts, predicted_species)):
                    container[key]["deleted"] += 1
            else:
                final_label = final_labels[final_index]
                final_species = normalize_species(final_label.get("species"))
                correct = predicted_species == final_species
                outcome = "unchanged_correct" if correct else "species_corrected"
                instance["predictions_retained"] += 1
                instance["species_correct"] += int(correct)
                instance["species_corrected"] += int(not correct)
                for container, key in ((tier_counts, tier), (band_counts, band), (species_counts, predicted_species)):
                    container[key]["retained"] += 1
                    container[key]["correct"] += int(correct)
                    container[key]["species_corrected"] += int(not correct)
            if outcome == "species_corrected":
                confusions[(predicted_species, final_species)] += 1
            instance_rows.append(
                {
                    "review_index": review_index,
                    "frame_key": frame_key,
                    "label_id": source_label.get("label_id", ""),
                    "predicted_species": predicted_species,
                    "final_species": final_species,
                    "correct": correct,
                    "outcome": outcome,
                    "tier": tier,
                    "confidence": prediction.get("confidence"),
                    "method": prediction.get("method", ""),
                }
            )

    retained_accuracy = safe_div(instance["species_correct"], instance["predictions_retained"])
    adjusted_accuracy = safe_div(instance["species_correct"], instance["predictions_before"])
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in instance_rows:
        if row["outcome"] == "deleted":
            continue
        predicted_species = str(row["predicted_species"])
        final_species = str(row["final_species"])
        class_counts[predicted_species]["predicted"] += 1
        class_counts[final_species]["actual"] += 1
        if row["correct"]:
            class_counts[final_species]["tp"] += 1
    class_metrics: dict[str, dict[str, Any]] = {}
    for name, counts in class_counts.items():
        precision = safe_div(counts["tp"], counts["predicted"])
        recall = safe_div(counts["tp"], counts["actual"])
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        class_metrics[name] = {
            "actual": counts["actual"],
            "predicted": counts["predicted"],
            "correct": counts["tp"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    evaluated_classes = [value for value in class_metrics.values() if value["actual"] > 0]
    macro_f1 = safe_div(sum(value["f1"] for value in evaluated_classes), len(evaluated_classes))
    weighted_f1 = safe_div(
        sum(value["f1"] * value["actual"] for value in evaluated_classes),
        sum(value["actual"] for value in evaluated_classes),
    )
    source_summary = load_json(source_dir / "summary.json")
    result = {
        "generated_on": date.today().isoformat(),
        "scope": {
            "sampling_spacing_m": cfg.spacing_m,
            "requested_first_frames": cfg.reviewed_frame_count,
            "reviewed_primary_frames": len(reviewed_keys),
            "missing_primary_frames": missing_keys,
            "formally_saved_frames_outside_primary": saved_outside_primary,
            "iou_matching_threshold": cfg.iou_threshold,
        },
        "frame_status": dict(status_counts),
        "before_after_species_prelabel": {
            "pre_review_predictions": instance["predictions_before"],
            "retained_after_review": instance["predictions_retained"],
            "deleted_during_review": instance["predictions_deleted"],
            "species_corrected_during_review": instance["species_corrected"],
            "unchanged_correct": instance["species_correct"],
            "accuracy_including_deleted_as_errors": adjusted_accuracy,
            "retained_only_top1_accuracy": retained_accuracy,
            "retained_only_macro_f1": macro_f1,
            "retained_only_weighted_f1": weighted_f1,
            "missing_pre_review_prediction": instance["missing_pre_review_prediction"],
        },
        "by_tier": {key: comparison_row(value) for key, value in sorted(tier_counts.items())},
        "by_confidence": {key: comparison_row(value) for key, value in band_counts.items()},
        "by_predicted_species": {
            key: comparison_row(value)
            for key, value in sorted(species_counts.items(), key=lambda item: (-item[1]["before"], item[0]))
        },
        "by_species": dict(
            sorted(
                class_metrics.items(),
                key=lambda item: (-item[1]["actual"], -item[1]["predicted"], item[0]),
            )
        ),
        "top_confusions": [
            {"predicted": predicted, "final": final, "count": count}
            for (predicted, final), count in confusions.most_common(20)
        ],
        "prior_independent_benchmark": source_summary.get("vmms_domain_classifier", {}),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{date.today():%Y%m%d}_VMMS识别精度报告_人工验收1-{cfg.reviewed_frame_count}帧"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}_逐实例.csv"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "review_index", "frame_key", "label_id", "predicted_species", "final_species",
            "correct", "outcome", "tier", "confidence", "method",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(instance_rows)

    tier_rows = [
        [
            name,
            values["before"],
            values["retained"],
            values["deleted"],
            values["species_corrected"],
            values["correct"],
            pct(values["accuracy_including_deleted"]),
        ]
        for name, values in result["by_tier"].items()
    ]
    band_order = ["≥0.99", "0.90–0.99", "0.70–0.90", "0.50–0.70", "<0.50", "无可比模型分数"]
    band_rows = [
        [
            name,
            result["by_confidence"][name]["before"],
            result["by_confidence"][name]["retained"],
            result["by_confidence"][name]["deleted"],
            result["by_confidence"][name]["species_corrected"],
            result["by_confidence"][name]["correct"],
            pct(result["by_confidence"][name]["accuracy_including_deleted"]),
        ]
        for name in band_order if name in result["by_confidence"]
    ]
    species_rows = [
        [
            name,
            values["before"],
            values["retained"],
            values["deleted"],
            values["species_corrected"],
            values["correct"],
            pct(values["accuracy_including_deleted"]),
        ]
        for name, values in result["by_predicted_species"].items()
        if values["before"] >= 5
    ]
    confusion_rows = [
        [row["predicted"], row["final"], row["count"]]
        for row in result["top_confusions"][:10]
    ]
    benchmark = result["prior_independent_benchmark"]
    markdown = f"""# 香港 VMMS 树种识别精度报告：验收前后对比

生成日期：{date.today().isoformat()}

## 结论摘要

- 主验收集：5 m 采样序列前 {cfg.reviewed_frame_count} 帧；已形成正式审核记录 {len(reviewed_keys)} 帧，尚未正式提交 {len(missing_keys)} 帧。
- 验收前自动树种预标 {instance['predictions_before']} 个；验收后保留 {instance['predictions_retained']} 个、删除 {instance['predictions_deleted']} 个。
- 保留实例中，原树种正确 {instance['species_correct']} 个、被人工改类 {instance['species_corrected']} 个；仅看保留实例的条件 Top-1 为 **{pct(retained_accuracy)}**。
- 按本次要求把删除实例也计作原识别错误后，验收前预标的总体准确率为 **{pct(adjusted_accuracy)}**（{instance['species_correct']}/{instance['predictions_before']}）。
- 保留实例的 Macro-F1 为 **{pct(macro_f1)}**，Weighted-F1 为 **{pct(weighted_f1)}**；这两个 F1 不纳入已删除实例。
- 独立路线块验证基线仍为 Top-1 **{pct(benchmark.get('held_out_top1_accuracy'))}**；置信度 ≥0.99 的选择性准确率 **{pct(benchmark.get('held_out_accuracy_at_confidence_0.99'))}**。

## 验收范围

本报告以当前逐帧验收工具中的正式记录为人工结果，严格按 5 m 采样顺序取第 1–{cfg.reviewed_frame_count} 帧。帧状态：有树 {status_counts['annotated']}、无树 {status_counts['no_tree']}、不可用 {status_counts['unusable']}。在主范围外正式保存的帧数为 {len(saved_outside_primary)}，不混入主指标。

细叶榕与垂叶榕在预测和人工结果两侧均先归并为“榕树”再计分。原有人工作为锚点的 905 个标签不当作模型树种预测；只有验收前实际存在的自动树种预标进入分母。人工删除的预标按错误计，人工新增且验收前不存在的实例不进入模型树种准确率。

## 验收前后总体变化

| 验收前预标 | 验收后保留 | 删除 | 人工改类 | 原标签正确 | 含删除总体准确率 | 仅保留 Top-1 |
|---|---|---|---|---|---|---|
| {instance['predictions_before']} | {instance['predictions_retained']} | {instance['predictions_deleted']} | {instance['species_corrected']} | {instance['species_correct']} | {pct(adjusted_accuracy)} | {pct(retained_accuracy)} |

## 分层对比

### 按助手证据等级

{markdown_table(['等级', '验收前', '保留', '删除', '改类', '正确', '含删除准确率'], tier_rows)}

### 按界面显示置信度

{markdown_table(['置信度', '验收前', '保留', '删除', '改类', '正确', '含删除准确率'], band_rows)}

### 按验收前预测类别（预标数 ≥5）

{markdown_table(['验收前预测类别', '验收前', '保留', '删除', '改类', '正确', '含删除准确率'], species_rows)}

### 主要混淆

{markdown_table(['模型预测', '人工结果', '数量'], confusion_rows) if confusion_rows else '本批没有错误混淆。'}

## 解释与限制

- 本报告不评价分割或检测精度。实例按相同 `label_id` 对齐；仅在 ID 因重画发生变化时，才用位置重叠关系寻找同一棵树。
- 相邻全景帧高度相关，因此没有把实例当作独立同分布样本计算置信区间。
- 应用户要求，审核中删除的自动预标一律计为原识别错误；如果个别删除实际是重复框或轮廓质量问题，这一规则会保守地下调树种识别准确率。
- 仅保留 Top-1 用来观察纯树种改类表现；含删除总体准确率才是本报告的主指标。
- 正式训练前仍应把审核后的路线块与训练路线完全隔离，生成独立测试集后再给出最终 Top-1、Macro-F1、混淆矩阵和置信度校准结果。

## 文件

- 逐实例审计：`{csv_path}`
- 机器可读汇总：`{json_path}`
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"report": str(markdown_path), "json": str(json_path), "csv": str(csv_path), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
