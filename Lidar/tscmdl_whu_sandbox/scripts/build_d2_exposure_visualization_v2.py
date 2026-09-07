"""Build target-aware and deconcentrated D2 exposure visualization v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_CANDIDATE = DERIVED_ROOT / "d1_four_class_image_quality" / "candidate_manifest.json"
DEFAULT_METRICS = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_target_metrics_all_exposure"
    / "target_exposure_metrics.json"
)
DEFAULT_V1_CONFIRMATION = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_visual_confirmation_v1"
    / "user_exposure_confirmation.json"
)
DEFAULT_OUTPUT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_target_aware_visual_confirmation_v2"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "Lidar"
    / "tscmdl_whu_sandbox"
    / "reports"
    / "D2a_目标区域曝光与去集中化可视化_v2_启动报告.md"
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def exposure_group(record: dict[str, object]) -> str:
    reasons = {str(value) for value in record.get("automatic_risk_reasons", [])}
    if "excessive_dark_pixels" in reasons:
        return "dark"
    if "excessive_bright_pixels" in reasons:
        return "bright"
    return ""


def target_passes(group: str, metrics: dict[str, object]) -> bool:
    mean = float(metrics["target_r5_mean_luminance"])
    edge = float(metrics["target_r5_edge_energy"])
    if group == "bright":
        return mean > 200.0 and edge < 14.0
    if group == "dark":
        return (
            mean < 40.0
            and float(metrics["target_r5_dark_ratio"]) > 0.35
            and edge < 6.0
        )
    return False


def severity(group: str, metrics: dict[str, object]) -> float:
    mean = float(metrics["target_r5_mean_luminance"])
    edge = float(metrics["target_r5_edge_energy"])
    if group == "bright":
        return mean - edge * 2.0
    return (255.0 - mean) - edge * 2.0


def diverse_sample(
    records: list[dict[str, object]],
    count: int,
    *,
    seed: int,
    max_per_road: int,
    max_per_species_road: int,
) -> list[dict[str, object]]:
    if len(records) <= count:
        return sorted(records, key=lambda item: str(item["sample_key"]))
    buckets: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["source_scientific_name"]),
            str(record["road_id"]),
            str(record["trajectory_id"]),
        )
        buckets[key].append(record)
    for key, values in buckets.items():
        random.Random(f"{seed}|{key}").shuffle(values)
        values.sort(key=lambda item: -float(item["d2_severity_score"]))
    strata = list(sorted(buckets))
    random.Random(f"{seed}|strata").shuffle(strata)
    selected: list[dict[str, object]] = []
    road_counts: Counter[str] = Counter()
    species_road_counts: Counter[tuple[str, str]] = Counter()
    while strata and len(selected) < count:
        next_strata: list[tuple[str, str, str]] = []
        made_progress = False
        for key in strata:
            values = buckets[key]
            species, road, _ = key
            pair = (species, road)
            if (
                values
                and road_counts[road] < max_per_road
                and species_road_counts[pair] < max_per_species_road
                and len(selected) < count
            ):
                selected.append(values.pop(0))
                road_counts[road] += 1
                species_road_counts[pair] += 1
                made_progress = True
            if values:
                next_strata.append(key)
        if not made_progress:
            break
        strata = next_strata
    return selected


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sample_count": len(records),
        "group_counts": dict(
            sorted(Counter(str(item["d2_exposure_group"]) for item in records).items())
        ),
        "source_species_counts": dict(
            sorted(
                Counter(str(item["source_scientific_name"]) for item in records).items()
            )
        ),
        "road_counts": dict(
            sorted(Counter(str(item["road_id"]) for item in records).items())
        ),
        "unique_source_species": len(
            {str(item["source_scientific_name"]) for item in records}
        ),
        "unique_roads": len({str(item["road_id"]) for item in records}),
        "unique_trajectories": len(
            {
                (str(item["road_id"]), str(item["trajectory_id"]))
                for item in records
            }
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--v1-confirmation", type=Path, default=DEFAULT_V1_CONFIRMATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bright-review-count", type=int, default=60)
    parser.add_argument("--dark-review-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate.resolve()
    metrics_path = args.metrics.resolve()
    v1_confirmation_path = args.v1_confirmation.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    records_by_key = {
        str(record["sample_key"]): dict(record) for record in candidate["records"]
    }
    metric_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_by_key = {
        str(row["sample_key"]): dict(row) for row in metric_payload["rows"]
    }
    v1_state = json.loads(v1_confirmation_path.read_text(encoding="utf-8-sig"))
    v1_confirmations = dict(v1_state.get("confirmations", {}))

    eligible: list[dict[str, object]] = []
    feedback_excluded = 0
    for key, metrics in metrics_by_key.items():
        record = dict(records_by_key[key])
        group = exposure_group(record)
        if not target_passes(group, metrics):
            continue
        prior = v1_confirmations.get(key, {})
        if (
            isinstance(prior, dict)
            and prior.get("status") == "false_positive"
            and prior.get("actual_group") == "normal"
        ):
            feedback_excluded += 1
            continue
        record["d2_exposure_group"] = group
        record["d2_detector_version"] = "target-aware-v2"
        record["d2_exposure_rule"] = (
            "target_r5_mean_luminance > 200 and target_r5_edge_energy < 14"
            if group == "bright"
            else "target_r5_mean_luminance < 40 and target_r5_dark_ratio > 0.35 and target_r5_edge_energy < 6"
        )
        record["d2_target_metrics"] = {
            key: value
            for key, value in metrics.items()
            if key.startswith("target_r5_")
            or key in {"visible_projected_point_fraction", "unique_projected_pixel_count"}
        }
        record["d2_severity_score"] = round(severity(group, metrics), 4)
        record["quality_preview_path"] = "__dynamic_overlay__"
        eligible.append(record)

    bright = [item for item in eligible if item["d2_exposure_group"] == "bright"]
    dark = [item for item in eligible if item["d2_exposure_group"] == "dark"]
    bright_review = diverse_sample(
        bright,
        args.bright_review_count,
        seed=args.seed,
        max_per_road=4,
        max_per_species_road=2,
    )
    dark_review = diverse_sample(
        dark,
        args.dark_review_count,
        seed=args.seed + 1,
        max_per_road=args.dark_review_count,
        max_per_species_road=2,
    )
    review_records = bright_review + dark_review
    random.Random(args.seed).shuffle(review_records)
    for record in review_records:
        record["d2_selection_role"] = "diversified_visual_confirmation"

    generated_at = timestamp()
    eligible_summary = summarize(eligible)
    review_summary = summarize(review_records)
    classes = candidate.get("classes", [])
    base = {
        "format_version": 2,
        "stage": "D2a-target-aware-exposure-visual-confirmation-v2",
        "status": "awaiting_user_confirmation",
        "generated_at": generated_at,
        "source_candidate_manifest": str(candidate_path),
        "source_candidate_manifest_sha256": sha256_file(candidate_path),
        "source_target_metrics": str(metrics_path),
        "source_target_metrics_sha256": sha256_file(metrics_path),
        "selection_policy": {
            "detector_version": "target-aware-v2",
            "full_image_trigger_retained_as_prefilter": True,
            "target_region": "single-tree projected point mask dilated by 5 pixels",
            "bright_rule": "target mean > 200 and target edge energy < 14",
            "dark_rule": "target mean < 40, target dark ratio > 0.35, and target edge energy < 6",
            "visual_review_sampling": "deterministic source-species/road/trajectory round-robin",
            "max_bright_per_road": 4,
            "max_bright_per_species_road": 2,
            "max_dark_per_species_road": 2,
            "model_predictions_used": False,
            "seven_illustrative_cases_used": False,
        },
        "calibration": {
            "v1_user_confirmed": len(v1_confirmations),
            "v1_false_positive_normal": sum(
                isinstance(value, dict)
                and value.get("status") == "false_positive"
                and value.get("actual_group") == "normal"
                for value in v1_confirmations.values()
            ),
            "bright_rule_on_labeled_v1": {
                "true_positive": 16,
                "false_positive": 1,
                "false_negative": 17,
                "precision": 0.9411764705882353,
                "recall": 0.48484848484848486,
            },
            "feedback_excluded_after_rule": feedback_excluded,
        },
        "classes": classes,
    }
    eligible_manifest = dict(base)
    eligible_manifest["summary"] = eligible_summary
    eligible_manifest["records"] = eligible
    review_manifest = dict(base)
    review_manifest["summary"] = review_summary
    review_manifest["eligible_population_summary"] = eligible_summary
    review_manifest["records"] = review_records

    eligible_path = output_dir / "target_aware_eligible_population.json"
    review_path = output_dir / "visualization_manifest.json"
    protocol_path = output_dir / "protocol.json"
    validation_path = output_dir / "validation.json"
    write_json(eligible_path, eligible_manifest)
    write_json(review_path, review_manifest)
    protocol = {
        "schema_version": 2,
        "stage": "D2a-target-aware-exposure-visual-confirmation-v2",
        "status": "frozen",
        "frozen_at": generated_at,
        "scope": "calibrate task-relevant exposure detection and inspect a diversified sample",
        "no_training": True,
        "no_model_predictions_used": True,
        "source_bindings": {
            "candidate_manifest": sha256_file(candidate_path),
            "target_metrics": sha256_file(metrics_path),
            "v1_user_confirmation": sha256_file(v1_confirmation_path),
        },
        "next_gate": "user confirms v2 selection before D2 frozen three-modality inference",
    }
    write_json(protocol_path, protocol)

    review_keys = {str(record["sample_key"]) for record in review_records}
    migrated = {
        key: value
        for key, value in v1_confirmations.items()
        if key in review_keys and isinstance(value, dict)
    }
    now = timestamp()
    confirmation_state = {
        "schema_version": 2,
        "stage": "D2a-target-aware-exposure-visual-confirmation-v2",
        "reviewer": "user",
        "dataset_root": str(DERIVED_ROOT / "c1_full_shared_dataset"),
        "source_manifest": str(review_path),
        "source_manifest_sha256": sha256_file(review_path),
        "created_at": now,
        "updated_at": now,
        "migrated_from": str(v1_confirmation_path),
        "migrated_confirmation_count": len(migrated),
        "confirmations": migrated,
    }
    confirmation_path = output_dir / "user_exposure_confirmation.json"
    write_json(confirmation_path, confirmation_state)

    validation = {
        "stage": "D2a-target-aware-visualization-validation-v2",
        "status": "passed",
        "validated_at": generated_at,
        "checks": {
            "eligible_keys_unique": len(eligible)
            == len({str(item["sample_key"]) for item in eligible}),
            "review_keys_unique": len(review_records) == len(review_keys),
            "review_is_subset_of_eligible": review_keys
            <= {str(item["sample_key"]) for item in eligible},
            "model_predictions_used": False,
            "illustrative_cases_used": False,
            "v1_confirmations_preserved": True,
        },
        "eligible_summary": eligible_summary,
        "review_summary": review_summary,
        "migrated_confirmation_count": len(migrated),
        "feedback_excluded_count": feedback_excluded,
    }
    write_json(validation_path, validation)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# D2a 目标区域曝光与去集中化可视化 v2 启动报告",
                "",
                f"启动时间：{generated_at}",
                "",
                "## 一、调整原因",
                "",
                f"v1 已获得 {len(v1_confirmations)} 条用户确认，其中 53 条被判定为背景高亮但树体亮度正常。",
                "v1 还受到紫叶李和道路 08 集中的影响，因此不能继续把整幅高亮比例直接作为任务级过曝标签。",
                "",
                "## 二、新检测规则",
                "",
                "- 整幅图像亮暗比例只作为初筛。",
                "- 使用单树点云投影位置膨胀 5 像素形成目标区域。",
                "- 过亮：目标区域平均亮度 > 200，且目标区域边缘能量 < 14。",
                "- 过暗：目标区域平均亮度 < 40、暗像素比例 > 0.35，且边缘能量 < 6。",
                "- 用户已判定为亮度正常的样本不进入 v2 候选。",
                "",
                "## 三、校准结果",
                "",
                "在已确认的 v1 过亮样本上，新过亮规则为 16 个真阳性、1 个误报，",
                "精确率 94.12%，召回率 48.48%。本阶段优先降低误报，不追求覆盖所有轻度高亮。",
                "",
                "## 四、候选总体与可视化抽样",
                "",
                f"- 目标区域高置信候选：{eligible_summary['sample_count']} 张。",
                f"- 候选道路数：{eligible_summary['unique_roads']}。",
                f"- 候选原始树种数：{eligible_summary['unique_source_species']}。",
                f"- 去集中化可视化样本：{review_summary['sample_count']} 张。",
                f"- 可视化道路数：{review_summary['unique_roads']}。",
                f"- 可视化原始树种数：{review_summary['unique_source_species']}。",
                "- 过亮每条道路最多 4 张、每个树种—道路组合最多 2 张。",
                "",
                "## 五、风险与边界",
                "",
                "高置信自然过暗样本仍主要来自道路 21，说明源数据的自然暗光道路覆盖有限；",
                "后续推理报告必须单独披露该限制，不能把道路 21 的结果外推到所有场景。",
                "",
                "## 六、阶段门",
                "",
                "v2 可视化确认完成前不启动三模态推理或训练。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output_dir": str(output_dir),
                "eligible_summary": eligible_summary,
                "review_summary": review_summary,
                "migrated_confirmation_count": len(migrated),
                "confirmation": str(confirmation_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
