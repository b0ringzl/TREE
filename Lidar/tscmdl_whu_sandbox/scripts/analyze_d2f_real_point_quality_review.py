"""Analyze completed D2f human point-quality reviews against frozen model outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "lidar data"
    / "whu"
    / "derived"
    / "tscmdl"
    / "d2_exposure_stratified_evaluation"
    / "20260821_real_point_quality_review_v1"
)
MODALITIES = ("image", "point", "fusion")
MODALITY_LABELS = {"image": "单影像", "point": "单点云", "fusion": "融合"}
COMPLETENESS_ORDER = ("complete", "slight_loss", "moderate_loss", "severe_loss")
COMPLETENESS_SCORE = {name: index for index, name in enumerate(COMPLETENESS_ORDER)}
ANALYSIS_GROUPS = ("complete", "slight_loss", "moderate_or_severe")
GROUP_LABELS = {"complete": "完整", "slight_loss": "轻微缺失", "moderate_or_severe": "中/严重缺失"}
RNG_SEED = 20260823


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson_ci(correct: int, total: int) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    interval = binomtest(correct, total).proportion_ci(confidence_level=0.95, method="wilson")
    return [float(interval.low), float(interval.high)]


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 5000) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def cluster_bootstrap_difference(
    rows: list[dict[str, object]], field: str, rng: np.random.Generator, draws: int = 5000
) -> list[float]:
    roads = sorted({str(row["road_id"]) for row in rows})
    by_road: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_road[str(row["road_id"])].append(row)
    differences = []
    for _ in range(draws):
        sampled_roads = rng.choice(roads, size=len(roads), replace=True)
        sampled = [row for road in sampled_roads for row in by_road[str(road)]]
        complete = [float(row[field]) for row in sampled if not bool(row["any_completeness_loss"])]
        loss = [float(row[field]) for row in sampled if bool(row["any_completeness_loss"])]
        if complete and loss:
            differences.append(float(np.mean(loss) - np.mean(complete)))
    if not differences:
        return [float("nan"), float("nan")]
    return [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]


def stratified_permutation_difference(
    rows: list[dict[str, object]], field: str, rng: np.random.Generator, draws: int = 20000
) -> tuple[float, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(bool(row["any_completeness_loss"])) for row in rows], dtype=np.int8)
    observed = float(values[labels == 1].mean() - values[labels == 0].mean())
    strata: dict[tuple[int, str], np.ndarray] = {}
    for key in sorted({(int(row["class_index"]), str(row["split"])) for row in rows}):
        strata[key] = np.asarray(
            [index for index, row in enumerate(rows) if (int(row["class_index"]), str(row["split"])) == key],
            dtype=np.int64,
        )
    exceed = 0
    working = labels.copy()
    for _ in range(draws):
        for indices in strata.values():
            working[indices] = rng.permutation(labels[indices])
        difference = float(values[working == 1].mean() - values[working == 0].mean())
        exceed += int(abs(difference) >= abs(observed) - 1e-15)
    return observed, float((exceed + 1) / (draws + 1))


def stratified_permutation_trend(
    rows: list[dict[str, object]], field: str, rng: np.random.Generator, draws: int = 20000
) -> tuple[float, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    severity = np.asarray([int(row["completeness_severity"]) for row in rows], dtype=np.float64)
    strata_indices = []
    for key in sorted({(int(row["class_index"]), str(row["split"])) for row in rows}):
        strata_indices.append(
            np.asarray(
                [index for index, row in enumerate(rows) if (int(row["class_index"]), str(row["split"])) == key],
                dtype=np.int64,
            )
        )
    residual_values = values.copy()
    residual_severity = severity.copy()
    for indices in strata_indices:
        residual_values[indices] -= residual_values[indices].mean()
        residual_severity[indices] -= residual_severity[indices].mean()

    def correlation(x: np.ndarray, y: np.ndarray) -> float:
        denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
        return 0.0 if denominator == 0 else float(np.dot(x, y) / denominator)

    observed = correlation(residual_severity, residual_values)
    exceed = 0
    working = severity.copy()
    for _ in range(draws):
        for indices in strata_indices:
            working[indices] = rng.permutation(severity[indices])
        residual_working = working.copy()
        for indices in strata_indices:
            residual_working[indices] -= residual_working[indices].mean()
        result = correlation(residual_working, residual_values)
        exceed += int(abs(result) >= abs(observed) - 1e-15)
    return observed, float((exceed + 1) / (draws + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * float(p_values[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def group_summary(
    rows: list[dict[str, object]], split: str, group: str, modality: str, rng: np.random.Generator
) -> dict[str, object]:
    selected = [
        row for row in rows
        if (split == "all" or str(row["split"]) == split) and str(row["analysis_group"]) == group
    ]
    true = [int(row["class_index"]) for row in selected]
    predicted = [int(row[f"{modality}_prediction"]) for row in selected]
    correct = int(sum(bool(row[f"{modality}_correct"]) for row in selected))
    probabilities = np.asarray([float(row[f"{modality}_true_probability"]) for row in selected], dtype=np.float64)
    return {
        "split": split,
        "analysis_group": group,
        "modality": modality,
        "tree_count": len(selected),
        "correct_count": correct,
        "accuracy": float(correct / len(selected)) if selected else float("nan"),
        "accuracy_wilson_95_ci": wilson_ci(correct, len(selected)),
        "macro_f1": float(f1_score(true, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0)) if selected else float("nan"),
        "mean_true_probability": float(probabilities.mean()) if len(probabilities) else float("nan"),
        "mean_true_probability_bootstrap_95_ci": bootstrap_mean_ci(probabilities, rng),
        "class_counts": dict(sorted(Counter(str(row["scientific_name"]) for row in selected).items())),
        "road_count": len({str(row["road_id"]) for row in selected}),
    }


def paired_fusion_point(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    selected = [row for row in rows if str(row["analysis_group"]) == group]
    fusion_only = sum(bool(row["fusion_correct"]) and not bool(row["point_correct"]) for row in selected)
    point_only = sum(bool(row["point_correct"]) and not bool(row["fusion_correct"]) for row in selected)
    discordant = fusion_only + point_only
    p_value = float(binomtest(fusion_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "analysis_group": group,
        "tree_count": len(selected),
        "fusion_correct_point_wrong": fusion_only,
        "point_correct_fusion_wrong": point_only,
        "fusion_net_correct_gain": fusion_only - point_only,
        "mcnemar_exact_two_sided_p": p_value,
    }


def plot_accuracy(summaries: list[dict[str, object]], path: Path) -> None:
    data = {(row["analysis_group"], row["modality"]): row for row in summaries if row["split"] == "all"}
    x = np.arange(len(ANALYSIS_GROUPS))
    width = 0.24
    colors = {"image": "#4472C4", "point": "#ED7D31", "fusion": "#70AD47"}
    fig, axis = plt.subplots(figsize=(10.8, 6.2))
    for index, modality in enumerate(MODALITIES):
        values = [100.0 * float(data[(group, modality)]["accuracy"]) for group in ANALYSIS_GROUPS]
        lower = [100.0 * float(data[(group, modality)]["accuracy_wilson_95_ci"][0]) for group in ANALYSIS_GROUPS]
        upper = [100.0 * float(data[(group, modality)]["accuracy_wilson_95_ci"][1]) for group in ANALYSIS_GROUPS]
        positions = x + (index - 1) * width
        axis.bar(positions, values, width, label=MODALITY_LABELS[modality], color=colors[modality], alpha=0.9)
        axis.errorbar(positions, values, yerr=[np.asarray(values) - lower, np.asarray(upper) - values], fmt="none", ecolor="#303030", capsize=3, lw=1)
        for px, value in zip(positions, values):
            axis.text(px, value + 2.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    counts = [int(data[(group, "point")]["tree_count"]) for group in ANALYSIS_GROUPS]
    axis.set_xticks(x, [f"{GROUP_LABELS[group]}\n(n={count})" for group, count in zip(ANALYSIS_GROUPS, counts)])
    axis.set_ylim(0, 112)
    axis.set_ylabel("准确率（%）")
    axis.set_title("D2f 真实点云完整度分层下的三模态准确率")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_true_probability(summaries: list[dict[str, object]], path: Path) -> None:
    data = {(row["analysis_group"], row["modality"]): row for row in summaries if row["split"] == "all"}
    x = np.arange(len(ANALYSIS_GROUPS))
    colors = {"image": "#4472C4", "point": "#ED7D31", "fusion": "#70AD47"}
    fig, axis = plt.subplots(figsize=(10.5, 6.0))
    for modality in MODALITIES:
        values = [float(data[(group, modality)]["mean_true_probability"]) for group in ANALYSIS_GROUPS]
        lower = [float(data[(group, modality)]["mean_true_probability_bootstrap_95_ci"][0]) for group in ANALYSIS_GROUPS]
        upper = [float(data[(group, modality)]["mean_true_probability_bootstrap_95_ci"][1]) for group in ANALYSIS_GROUPS]
        axis.errorbar(x, values, yerr=[np.asarray(values) - lower, np.asarray(upper) - values], marker="o", lw=2.2, capsize=4, label=MODALITY_LABELS[modality], color=colors[modality])
    axis.set_xticks(x, [GROUP_LABELS[group] for group in ANALYSIS_GROUPS])
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("真实类别平均概率")
    axis.set_title("D2f 点云完整度与模型真实类别证据")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(rows: list[dict[str, object]], path: Path) -> None:
    classes = sorted({str(row["scientific_name"]) for row in rows})
    colors = {"complete": "#70AD47", "slight_loss": "#FFC000", "moderate_loss": "#ED7D31", "severe_loss": "#C00000"}
    fig, axis = plt.subplots(figsize=(11.0, 6.2))
    bottom = np.zeros(len(classes), dtype=np.float64)
    for grade in COMPLETENESS_ORDER:
        values = []
        for class_name in classes:
            class_rows = [row for row in rows if str(row["scientific_name"]) == class_name]
            values.append(100.0 * sum(str(row["completeness"]) == grade for row in class_rows) / len(class_rows))
        axis.bar(np.arange(len(classes)), values, bottom=bottom, label=grade, color=colors[grade])
        bottom += np.asarray(values)
    axis.set_xticks(np.arange(len(classes)), classes, rotation=12, ha="right")
    axis.set_ylabel("该树种样本比例（%）")
    axis.set_ylim(0, 100)
    axis.set_title("D2f 人工完整度等级的树种分布")
    axis.legend(["完整", "轻微缺失", "中度缺失", "严重缺失"], ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_split_accuracy(summaries: list[dict[str, object]], path: Path) -> None:
    data = {(row["split"], row["analysis_group"], row["modality"]): row for row in summaries}
    colors = {"point": "#ED7D31", "fusion": "#70AD47"}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    x = np.arange(len(ANALYSIS_GROUPS))
    width = 0.34
    for axis, split in zip(axes, ("val", "test")):
        for index, modality in enumerate(("point", "fusion")):
            values = [100.0 * float(data[(split, group, modality)]["accuracy"]) for group in ANALYSIS_GROUPS]
            positions = x + (index - 0.5) * width
            axis.bar(positions, values, width, color=colors[modality], label=MODALITY_LABELS[modality])
            for px, value in zip(positions, values):
                axis.text(px, value + 1.6, f"{value:.1f}%", ha="center", fontsize=9)
        counts = [int(data[(split, group, "point")]["tree_count"]) for group in ANALYSIS_GROUPS]
        axis.set_xticks(x, [f"{GROUP_LABELS[group]}\n(n={count})" for group, count in zip(ANALYSIS_GROUPS, counts)])
        axis.set_ylim(0, 110)
        axis.set_title("验证集" if split == "val" else "测试集")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(loc="lower left")
    axes[0].set_ylabel("准确率（%）")
    fig.suptitle("D2f 验证集与测试集的点云完整度分层结果")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis_root = root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    review_path = root / "review_state.json"
    manifest_hash = sha256_file(manifest_path)
    review_hash = sha256_file(review_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    review_state = json.loads(review_path.read_text(encoding="utf-8-sig"))
    reviews = dict(review_state.get("reviews", {}))
    records_by_key = {str(row["sample_key"]): row for row in manifest["records"]}
    if set(reviews) != set(records_by_key):
        missing = sorted(set(records_by_key) - set(reviews))
        extra = sorted(set(reviews) - set(records_by_key))
        raise ValueError(f"review coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    lock_path = analysis_root / "locked_review_state.json"
    lock_meta_path = analysis_root / "review_lock.json"
    if lock_meta_path.exists():
        existing = json.loads(lock_meta_path.read_text(encoding="utf-8-sig"))
        if existing.get("review_state_sha256") != review_hash:
            raise ValueError("review_state.json changed after the D2f lock was created")
    else:
        shutil.copy2(review_path, lock_path)
        lock_payload = {
            "locked_at": timestamp(),
            "review_state": str(review_path),
            "review_state_sha256": review_hash,
            "locked_copy": str(lock_path),
            "locked_copy_sha256": sha256_file(lock_path),
            "manifest_sha256": manifest_hash,
            "review_count": len(reviews),
        }
        atomic_json(lock_meta_path, lock_payload)

    joined: list[dict[str, object]] = []
    for key, record in records_by_key.items():
        review = reviews[key]
        completeness = str(review["completeness"])
        if completeness not in COMPLETENESS_SCORE:
            raise ValueError(f"unsupported completeness label for {key}: {completeness}")
        severity = COMPLETENESS_SCORE[completeness]
        analysis_group = completeness if severity <= 1 else "moderate_or_severe"
        row: dict[str, object] = {
            "sample_key": key,
            "split": str(record["split"]),
            "class_index": int(record["class_index"]),
            "scientific_name": str(record["scientific_name"]),
            "road_id": str(record["road_id"]),
            "trajectory_id": str(record["trajectory_id"]),
            "tree_id": int(record["tree_id"]),
            "completeness": completeness,
            "completeness_severity": severity,
            "any_completeness_loss": bool(severity > 0),
            "analysis_group": analysis_group,
            "purity": str(review["purity"]),
            "overall_usability": str(review["overall_usability"]),
            "confidence": str(review["confidence"]),
            "issue_types": ";".join(review.get("issue_types", [])),
            "note": str(review.get("note", "")),
            "automatic_point_quality_risk_score": float(record["automatic_point_quality_risk_score"]),
            "source_point_count": int(record["point_metrics"]["source_point_count"]),
            "actual_sampled_unique_point_count": int(record["point_metrics"]["actual_sampled_unique_point_count"]),
            "repeat_fraction": float(record["point_metrics"]["repeat_fraction"]),
        }
        for modality in MODALITIES:
            outcome = record["model_outcomes"][modality]
            for field in ("prediction", "correct", "true_probability", "confidence", "entropy_normalized", "seed_agreement"):
                row[f"{modality}_{field}"] = outcome[field]
        joined.append(row)
    joined.sort(key=lambda row: int(records_by_key[str(row["sample_key"])]["review_order"]))
    write_csv(analysis_root / "d2f_joined_records.csv", joined)

    rng = np.random.default_rng(RNG_SEED)
    summaries = [
        group_summary(joined, split, group, modality, rng)
        for split in ("all", "val", "test")
        for group in ANALYSIS_GROUPS
        for modality in MODALITIES
    ]
    summary_csv_rows = []
    for row in summaries:
        summary_csv_rows.append(
            {
                "split": row["split"],
                "analysis_group": row["analysis_group"],
                "modality": row["modality"],
                "tree_count": row["tree_count"],
                "correct_count": row["correct_count"],
                "accuracy": row["accuracy"],
                "accuracy_ci_low": row["accuracy_wilson_95_ci"][0],
                "accuracy_ci_high": row["accuracy_wilson_95_ci"][1],
                "macro_f1": row["macro_f1"],
                "mean_true_probability": row["mean_true_probability"],
                "true_probability_ci_low": row["mean_true_probability_bootstrap_95_ci"][0],
                "true_probability_ci_high": row["mean_true_probability_bootstrap_95_ci"][1],
                "road_count": row["road_count"],
            }
        )
    write_csv(analysis_root / "d2f_quality_modality_summary.csv", summary_csv_rows)

    comparisons: dict[str, dict[str, object]] = {}
    accuracy_p: dict[str, float] = {}
    probability_p: dict[str, float] = {}
    trend_accuracy_p: dict[str, float] = {}
    trend_probability_p: dict[str, float] = {}
    for modality in MODALITIES:
        accuracy_field = f"{modality}_correct"
        probability_field = f"{modality}_true_probability"
        accuracy_difference, accuracy_perm_p = stratified_permutation_difference(joined, accuracy_field, rng)
        probability_difference, probability_perm_p = stratified_permutation_difference(joined, probability_field, rng)
        trend_accuracy, trend_accuracy_perm_p = stratified_permutation_trend(joined, accuracy_field, rng)
        trend_probability, trend_probability_perm_p = stratified_permutation_trend(joined, probability_field, rng)
        accuracy_p[modality] = accuracy_perm_p
        probability_p[modality] = probability_perm_p
        trend_accuracy_p[modality] = trend_accuracy_perm_p
        trend_probability_p[modality] = trend_probability_perm_p
        comparisons[modality] = {
            "any_loss_minus_complete_accuracy": accuracy_difference,
            "accuracy_road_cluster_bootstrap_95_ci": cluster_bootstrap_difference(joined, accuracy_field, rng),
            "accuracy_class_split_stratified_permutation_p": accuracy_perm_p,
            "any_loss_minus_complete_true_probability": probability_difference,
            "true_probability_road_cluster_bootstrap_95_ci": cluster_bootstrap_difference(joined, probability_field, rng),
            "true_probability_class_split_stratified_permutation_p": probability_perm_p,
            "ordinal_severity_partial_correlation_with_accuracy": trend_accuracy,
            "ordinal_accuracy_class_split_stratified_permutation_p": trend_accuracy_perm_p,
            "ordinal_severity_partial_correlation_with_true_probability": trend_probability,
            "ordinal_probability_class_split_stratified_permutation_p": trend_probability_perm_p,
        }
    accuracy_holm = holm_adjust(accuracy_p)
    probability_holm = holm_adjust(probability_p)
    trend_accuracy_holm = holm_adjust(trend_accuracy_p)
    trend_probability_holm = holm_adjust(trend_probability_p)
    for modality in MODALITIES:
        comparisons[modality]["accuracy_holm_adjusted_p_across_modalities"] = accuracy_holm[modality]
        comparisons[modality]["true_probability_holm_adjusted_p_across_modalities"] = probability_holm[modality]
        comparisons[modality]["ordinal_accuracy_holm_adjusted_p_across_modalities"] = trend_accuracy_holm[modality]
        comparisons[modality]["ordinal_probability_holm_adjusted_p_across_modalities"] = trend_probability_holm[modality]

    split_comparisons: dict[str, dict[str, dict[str, object]]] = {}
    for split in ("val", "test"):
        subset = [row for row in joined if str(row["split"]) == split]
        split_accuracy_p: dict[str, float] = {}
        split_probability_p: dict[str, float] = {}
        split_result: dict[str, dict[str, object]] = {}
        for modality in MODALITIES:
            accuracy_field = f"{modality}_correct"
            probability_field = f"{modality}_true_probability"
            accuracy_difference, accuracy_perm_p = stratified_permutation_difference(subset, accuracy_field, rng)
            probability_difference, probability_perm_p = stratified_permutation_difference(subset, probability_field, rng)
            split_accuracy_p[modality] = accuracy_perm_p
            split_probability_p[modality] = probability_perm_p
            split_result[modality] = {
                "any_loss_minus_complete_accuracy": accuracy_difference,
                "accuracy_road_cluster_bootstrap_95_ci": cluster_bootstrap_difference(subset, accuracy_field, rng),
                "accuracy_class_stratified_permutation_p": accuracy_perm_p,
                "any_loss_minus_complete_true_probability": probability_difference,
                "true_probability_road_cluster_bootstrap_95_ci": cluster_bootstrap_difference(subset, probability_field, rng),
                "true_probability_class_stratified_permutation_p": probability_perm_p,
            }
        split_accuracy_holm = holm_adjust(split_accuracy_p)
        split_probability_holm = holm_adjust(split_probability_p)
        for modality in MODALITIES:
            split_result[modality]["accuracy_holm_adjusted_p_across_modalities"] = split_accuracy_holm[modality]
            split_result[modality]["true_probability_holm_adjusted_p_across_modalities"] = split_probability_holm[modality]
        split_comparisons[split] = split_result

    risk = np.asarray([float(row["automatic_point_quality_risk_score"]) for row in joined])
    loss_binary = np.asarray([int(bool(row["any_completeness_loss"])) for row in joined])
    nonpass_binary = np.asarray([int(str(row["overall_usability"]) != "pass") for row in joined])
    severity = np.asarray([int(row["completeness_severity"]) for row in joined])
    risk_validation = {
        "any_completeness_loss_count": int(loss_binary.sum()),
        "any_loss_roc_auc": float(roc_auc_score(loss_binary, risk)),
        "any_loss_average_precision": float(average_precision_score(loss_binary, risk)),
        "nonpass_count": int(nonpass_binary.sum()),
        "nonpass_roc_auc": float(roc_auc_score(nonpass_binary, risk)),
        "nonpass_average_precision": float(average_precision_score(nonpass_binary, risk)),
        "risk_severity_spearman_rho": float(spearmanr(risk, severity).statistic),
        "risk_severity_spearman_p": float(spearmanr(risk, severity).pvalue),
    }
    issue_counts = Counter(issue for row in joined for issue in str(row["issue_types"]).split(";") if issue)
    by_class_grade = {
        class_name: dict(
            Counter(str(row["completeness"]) for row in joined if str(row["scientific_name"]) == class_name)
        )
        for class_name in sorted({str(row["scientific_name"]) for row in joined})
    }
    result = {
        "format_version": 1,
        "status": "complete",
        "generated_at": timestamp(),
        "design": "full D1 validation/test census with outcomes hidden until all human point-quality reviews were locked",
        "interpretation_limit": manifest.get("interpretation_limit"),
        "review_lock": json.loads(lock_meta_path.read_text(encoding="utf-8-sig")),
        "sample_count": len(joined),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in joined).items())),
        "completeness_counts": dict(Counter(str(row["completeness"]) for row in joined)),
        "analysis_group_counts": dict(Counter(str(row["analysis_group"]) for row in joined)),
        "purity_counts": dict(Counter(str(row["purity"]) for row in joined)),
        "usability_counts": dict(Counter(str(row["overall_usability"]) for row in joined)),
        "confidence_counts": dict(Counter(str(row["confidence"]) for row in joined)),
        "issue_counts": dict(issue_counts.most_common()),
        "by_class_completeness": by_class_grade,
        "quality_modality_summaries": summaries,
        "any_loss_vs_complete_comparisons": comparisons,
        "split_any_loss_vs_complete_comparisons": split_comparisons,
        "paired_fusion_vs_point": [paired_fusion_point(joined, group) for group in ANALYSIS_GROUPS],
        "automatic_risk_validation": risk_validation,
        "multiple_comparison_adjustment": "Holm correction across the three modalities within each predefined outcome family",
    }
    summary_path = analysis_root / "d2f_real_point_quality_summary.json"
    atomic_json(summary_path, result)
    plot_accuracy(summaries, analysis_root / "d2f_accuracy_by_completeness.png")
    plot_true_probability(summaries, analysis_root / "d2f_true_probability_by_completeness.png")
    plot_distribution(joined, analysis_root / "d2f_completeness_class_distribution.png")
    plot_split_accuracy(summaries, analysis_root / "d2f_split_accuracy_by_completeness.png")

    validation = {
        "status": "passed",
        "generated_at": timestamp(),
        "review_count": len(reviews),
        "joined_record_count": len(joined),
        "all_manifest_records_reviewed": set(reviews) == set(records_by_key),
        "all_required_review_fields_present": all(
            all(review.get(field) for field in ("completeness", "purity", "overall_usability", "confidence"))
            for review in reviews.values()
        ),
        "review_state_sha256": review_hash,
        "locked_copy_sha256": sha256_file(lock_path),
        "lock_hash_matches": sha256_file(lock_path) == review_hash,
        "prediction_modalities_present": all(
            all(f"{modality}_correct" in row for modality in MODALITIES) for row in joined
        ),
        "purity_has_variation": len({str(row["purity"]) for row in joined}) > 1,
        "summary_sha256": sha256_file(summary_path),
    }
    if not all(
        [
            validation["all_manifest_records_reviewed"],
            validation["all_required_review_fields_present"],
            validation["lock_hash_matches"],
            validation["prediction_modalities_present"],
        ]
    ):
        validation["status"] = "failed"
    atomic_json(analysis_root / "validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
