#!/usr/bin/env python3
"""Paired D2 exposure-case versus target-normal-control analysis.

The controls are different trees matched without replacement.  This script treats
the comparison as an observational matched analysis, not a same-tree exposure
intervention or a causal counterfactual.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


MODALITIES = ("image", "point", "fusion")
CLASS_NAMES = (
    "Cinnamomum camphora",
    "Lagerstroemia indica",
    "Magnolia grandiflora",
    "Other",
)
COLORS = {"image": "#E69F00", "point": "#0072B2", "fusion": "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_by_key(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_key"]: row for row in csv.DictReader(handle)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar_p(normal_only: int, abnormal_only: int) -> float:
    discordant = normal_only + abnormal_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(normal_only, abnormal_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_ci(values: Iterable[float], *, seed: int, repetitions: int = 20000) -> list[float]:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.size == 0:
        return [float("nan"), float("nan")]
    if vector.size == 1:
        value = float(vector[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    batch = 1000
    for start in range(0, repetitions, batch):
        stop = min(start + batch, repetitions)
        indices = rng.integers(0, vector.size, size=(stop - start, vector.size))
        means[start:stop] = vector[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def macro_f1(rows: list[dict[str, Any]], prefix: str) -> float:
    scores: list[float] = []
    for class_name in CLASS_NAMES:
        tp = sum(r[prefix + "_prediction"] == class_name and r["model_class_name"] == class_name for r in rows)
        fp = sum(r[prefix + "_prediction"] == class_name and r["model_class_name"] != class_name for r in rows)
        fn = sum(r[prefix + "_prediction"] != class_name and r["model_class_name"] == class_name for r in rows)
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def summarize(rows: list[dict[str, Any]], group_name: str, seed_offset: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "group": group_name,
        "pair_count": len(rows),
        "case_exposure_counts": dict(Counter(r["case_exposure_group"] for r in rows)),
        "class_counts": dict(Counter(r["model_class_name"] for r in rows)),
        "modalities": {},
    }
    for modal_index, modality in enumerate(MODALITIES):
        abnormal_correct = np.asarray([r[f"case_{modality}_correct"] for r in rows], dtype=np.float64)
        normal_correct = np.asarray([r[f"control_{modality}_correct"] for r in rows], dtype=np.float64)
        abnormal_prob = np.asarray([r[f"case_{modality}_true_probability"] for r in rows], dtype=np.float64)
        normal_prob = np.asarray([r[f"control_{modality}_true_probability"] for r in rows], dtype=np.float64)
        normal_only = int(np.sum((normal_correct == 1) & (abnormal_correct == 0)))
        abnormal_only = int(np.sum((normal_correct == 0) & (abnormal_correct == 1)))
        accuracy_deltas = abnormal_correct - normal_correct
        probability_deltas = abnormal_prob - normal_prob
        result["modalities"][modality] = {
            "abnormal_correct_count": int(abnormal_correct.sum()),
            "abnormal_accuracy": float(abnormal_correct.mean()) if len(rows) else float("nan"),
            "normal_correct_count": int(normal_correct.sum()),
            "normal_accuracy": float(normal_correct.mean()) if len(rows) else float("nan"),
            "abnormal_minus_normal_accuracy": float(accuracy_deltas.mean()) if len(rows) else float("nan"),
            "accuracy_delta_paired_bootstrap_95_ci": bootstrap_ci(
                accuracy_deltas,
                seed=20260819 + seed_offset * 10 + modal_index,
            ),
            "normal_correct_abnormal_wrong": normal_only,
            "normal_wrong_abnormal_correct": abnormal_only,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(normal_only, abnormal_only),
            "abnormal_macro_f1": macro_f1(rows, f"case_{modality}"),
            "normal_macro_f1": macro_f1(rows, f"control_{modality}"),
            "abnormal_mean_true_probability": float(abnormal_prob.mean()) if len(rows) else float("nan"),
            "normal_mean_true_probability": float(normal_prob.mean()) if len(rows) else float("nan"),
            "abnormal_minus_normal_mean_true_probability": float(probability_deltas.mean()) if len(rows) else float("nan"),
            "true_probability_delta_paired_bootstrap_95_ci": bootstrap_ci(
                probability_deltas,
                seed=202608190 + seed_offset * 10 + modal_index,
            ),
        }
    return result


def merge_rows(case_root: Path, control_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs_doc = load_json(control_root / "pairs.json")
    case_fusion_doc = load_json(case_root / "fusion" / "three_modality_comparison.json")
    control_fusion_doc = load_json(control_root / "fusion" / "three_modality_comparison.json")
    case_fusion = {row["sample_key"]: row for row in case_fusion_doc["rows"]}
    control_fusion = {row["sample_key"]: row for row in control_fusion_doc["rows"]}

    probability_tables: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for cohort, root in (("case", case_root), ("control", control_root)):
        for modality in ("image", "point"):
            probability_tables[(cohort, modality)] = load_csv_by_key(root / modality / "per_tree_summary.csv")

    rows: list[dict[str, Any]] = []
    for pair in pairs_doc["pairs"]:
        case_key = pair["case_sample_key"]
        control_key = pair["control_sample_key"]
        case = case_fusion[case_key]
        control = control_fusion[control_key]
        row: dict[str, Any] = dict(pair)
        row["case_model_class_name"] = case["true_species"]
        row["control_model_class_name"] = control["true_species"]
        row["exact_source_species_match"] = int(pair["case_source_species"] == pair["control_source_species"])
        row["same_road_match"] = int(pair["case_road_id"] == pair["control_road_id"])
        for cohort, source, sample_key in (("case", case, case_key), ("control", control, control_key)):
            for modality in MODALITIES:
                row[f"{cohort}_{modality}_prediction"] = source[f"{modality}_prediction"]
                row[f"{cohort}_{modality}_correct"] = int(source[f"{modality}_correct"])
                row[f"{cohort}_{modality}_confidence"] = float(source[f"{modality}_confidence"])
                if modality == "fusion":
                    probability = source["fusion_true_class_probability"]
                else:
                    probability = probability_tables[(cohort, modality)][sample_key]["ensemble_true_class_probability"]
                row[f"{cohort}_{modality}_true_probability"] = float(probability)
        rows.append(row)
    return rows, pairs_doc


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(summary_by_group: dict[str, dict[str, Any]], output: Path) -> None:
    groups = ("overall", "bright", "dark")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    x = np.arange(len(MODALITIES))
    width = 0.34
    for axis, group in zip(axes, groups):
        summary = summary_by_group[group]
        normal = [summary["modalities"][m]["normal_accuracy"] * 100 for m in MODALITIES]
        abnormal = [summary["modalities"][m]["abnormal_accuracy"] * 100 for m in MODALITIES]
        normal_bars = axis.bar(x - width / 2, normal, width, label="Matched normal", color="#A6CEE3")
        abnormal_bars = axis.bar(x + width / 2, abnormal, width, label="Exposure abnormal", color="#FB9A99")
        axis.bar_label(normal_bars, fmt="%.1f", padding=2, fontsize=8)
        axis.bar_label(abnormal_bars, fmt="%.1f", padding=2, fontsize=8)
        axis.set_title(f"{group.title()} (n={summary['pair_count']})")
        axis.set_xticks(x, [m.title() for m in MODALITIES])
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("D2c matched observational comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta(summary_by_group: dict[str, dict[str, Any]], output: Path) -> None:
    groups = ("overall", "bright", "dark")
    x = np.arange(len(groups))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    for idx, modality in enumerate(MODALITIES):
        values = [summary_by_group[g]["modalities"][modality]["abnormal_minus_normal_accuracy"] * 100 for g in groups]
        bars = axis.bar(x + (idx - 1) * width, values, width, label=modality.title(), color=COLORS[modality])
        axis.bar_label(bars, fmt="%+.1f", padding=2, fontsize=8)
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(x, [f"{g.title()}\n(n={summary_by_group[g]['pair_count']})" for g in groups])
    axis.set_ylabel("Abnormal minus matched-normal accuracy (pp)")
    axis.set_title("Paired accuracy direction; negative means lower on abnormal cases")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_transitions(summary: dict[str, Any], output: Path) -> None:
    labels = ["Normal correct / abnormal wrong", "Normal wrong / abnormal correct"]
    x = np.arange(len(MODALITIES))
    width = 0.34
    lost = [summary["modalities"][m]["normal_correct_abnormal_wrong"] for m in MODALITIES]
    gained = [summary["modalities"][m]["normal_wrong_abnormal_correct"] for m in MODALITIES]
    fig, axis = plt.subplots(figsize=(8.8, 4.6))
    bars1 = axis.bar(x - width / 2, lost, width, label=labels[0], color="#D55E00")
    bars2 = axis.bar(x + width / 2, gained, width, label=labels[1], color="#56B4E9")
    axis.bar_label(bars1, padding=2)
    axis.bar_label(bars2, padding=2)
    axis.set_xticks(x, [m.title() for m in MODALITIES])
    axis.set_ylabel("Paired tree count")
    axis.set_title("Overall paired correctness transitions")
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, pairs_doc = merge_rows(args.case_root, args.control_root)

    group_filters = {
        "overall": lambda r: True,
        "bright": lambda r: r["case_exposure_group"] == "bright",
        "dark": lambda r: r["case_exposure_group"] == "dark",
        "named_target_classes": lambda r: r["model_class_name"] != "Other",
        "other_class": lambda r: r["model_class_name"] == "Other",
        "exact_source_species": lambda r: bool(r["exact_source_species_match"]),
        "same_road": lambda r: bool(r["same_road_match"]),
        "exact_species_same_road": lambda r: bool(r["exact_source_species_match"] and r["same_road_match"]),
        "tier_0_same_species_same_road": lambda r: r["match_tier"] == 0,
        "tier_1_same_species_other_road": lambda r: r["match_tier"] == 1,
        "tier_2_same_class_same_road": lambda r: r["match_tier"] == 2,
    }
    summaries = {
        name: summarize([r for r in rows if predicate(r)], name, index)
        for index, (name, predicate) in enumerate(group_filters.items())
    }

    result = {
        "format_version": 1,
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "design": "one-to-one observational matching without replacement; controls are different trees",
        "causal_limit": "This is not a same-tree controlled exposure intervention. Residual road, tree, background, season, and acquisition confounding remains.",
        "pair_count": len(rows),
        "match_distribution": dict(Counter(r["match_tier_name"] for r in rows)),
        "exact_source_species_pair_count": sum(r["exact_source_species_match"] for r in rows),
        "same_road_pair_count": sum(r["same_road_match"] for r in rows),
        "summaries": summaries,
    }
    summary_path = args.output_dir / "d2c_paired_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(args.output_dir / "d2c_per_pair.csv", rows)

    flat_rows: list[dict[str, Any]] = []
    for group, group_summary in summaries.items():
        for modality, metrics in group_summary["modalities"].items():
            flat_rows.append({"group": group, "pair_count": group_summary["pair_count"], "modality": modality, **metrics})
    write_csv(args.output_dir / "d2c_subgroup_summary.csv", flat_rows)

    plot_accuracy(summaries, args.output_dir / "d2c_matched_accuracy.png")
    plot_delta(summaries, args.output_dir / "d2c_paired_accuracy_delta.png")
    plot_transitions(summaries["overall"], args.output_dir / "d2c_paired_transitions.png")

    validation = {
        "status": "passed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pair_count": len(rows),
        "expected_pair_count": pairs_doc["pair_count"],
        "unique_case_count": len({r["case_sample_key"] for r in rows}),
        "unique_control_count": len({r["control_sample_key"] for r in rows}),
        "all_pair_classes_consistent": all(
            r["model_class_name"] == r["case_model_class_name"] == r["control_model_class_name"]
            for r in rows
        ),
        "all_correctness_binary": all(
            r[f"{cohort}_{modality}_correct"] in (0, 1)
            for r in rows
            for cohort in ("case", "control")
            for modality in MODALITIES
        ),
        "all_probabilities_in_range": all(
            0.0 <= r[f"{cohort}_{modality}_true_probability"] <= 1.0
            for r in rows
            for cohort in ("case", "control")
            for modality in MODALITIES
        ),
        "summary_sha256": sha256(summary_path),
    }
    validation["status"] = "passed" if (
        validation["pair_count"] == validation["expected_pair_count"]
        and validation["unique_case_count"] == validation["expected_pair_count"]
        and validation["unique_control_count"] == validation["expected_pair_count"]
        and validation["all_pair_classes_consistent"]
        and validation["all_correctness_binary"]
        and validation["all_probabilities_in_range"]
    ) else "failed"
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
