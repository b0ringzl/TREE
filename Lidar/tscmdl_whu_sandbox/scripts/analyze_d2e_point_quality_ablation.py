#!/usr/bin/env python3
"""Analyze D2e same-tree point segmentation quality ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_key"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def bootstrap_ci(values: Iterable[float], seed: int, repetitions: int = 20000) -> list[float]:
    vector = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 1000):
        stop = min(start + 1000, repetitions)
        indices = rng.integers(0, len(vector), size=(stop - start, len(vector)))
        means[start:stop] = vector[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def macro_f1(rows: list[dict[str, Any]], modality: str) -> float:
    scores = []
    for class_name in CLASS_NAMES:
        tp = sum(r["true_species"] == class_name and r[f"{modality}_prediction"] == class_name for r in rows)
        fp = sum(r["true_species"] != class_name and r[f"{modality}_prediction"] == class_name for r in rows)
        fn = sum(r["true_species"] == class_name and r[f"{modality}_prediction"] != class_name for r in rows)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def merge(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(root / "manifest.json")
    image = load_csv(root / "image" / "per_tree_summary.csv")
    point = load_csv(root / "point" / "per_tree_summary.csv")
    fusion_doc = load_json(root / "fusion" / "three_modality_comparison.json")
    fusion = {str(row["sample_key"]): row for row in fusion_doc["rows"]}
    protocol_by_base = {
        str(item["base_sample_key"]): item for item in manifest["base_protocol"]
    }
    rows: list[dict[str, Any]] = []
    for record in manifest["records"]:
        key = str(record["sample_key"])
        image_row = image[key]
        point_row = point[key]
        fusion_row = fusion[key]
        protocol = protocol_by_base[str(record["base_sample_key"])]
        row: dict[str, Any] = {
            "sample_key": key,
            "base_sample_key": str(record["base_sample_key"]),
            "tree_id": int(record["tree_id"]),
            "road_id": str(record["road_id"]),
            "true_species": str(record["scientific_name"]),
            "source_scientific_name": str(record["source_scientific_name"]),
            "condition": str(record["point_quality_condition"]),
            "condition_order": int(record["point_quality_condition_order"]),
            "factor": str(record["point_quality_factor"]),
            "level": float(record["point_quality_level"]),
            "retained_target_fraction": float(record["retained_target_fraction"]),
            "contamination_fraction": float(record["contamination_fraction"]),
            "unique_output_point_count": int(record["point_quality_metrics"]["unique_output_point_count"]),
            "donor_base_sample_key": str(record.get("donor_base_sample_key", "")),
            "donor_model_class_name": str(record.get("donor_model_class_name", "")),
            "donor_same_road": int(bool(protocol["donor_same_road"])) if record["point_quality_factor"] == "impurity" else 0,
        }
        for modality, source in (("image", image_row), ("point", point_row)):
            row[f"{modality}_prediction"] = str(source["ensemble_predicted_species"])
            row[f"{modality}_correct"] = int(source["ensemble_correct"])
            row[f"{modality}_confidence"] = float(source["ensemble_confidence"])
            row[f"{modality}_true_probability"] = float(source["ensemble_true_class_probability"])
        row["fusion_prediction"] = str(fusion_row["fusion_prediction"])
        row["fusion_correct"] = int(fusion_row["fusion_correct"])
        row["fusion_confidence"] = float(fusion_row["fusion_confidence"])
        row["fusion_true_probability"] = float(fusion_row["fusion_true_class_probability"])
        rows.append(row)
    rows.sort(key=lambda row: (row["base_sample_key"], row["condition_order"]))
    return rows, manifest


def condition_summary(
    rows: list[dict[str, Any]],
    baseline_by_base: dict[str, dict[str, Any]],
    seed_offset: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "condition": rows[0]["condition"],
        "factor": rows[0]["factor"],
        "level": rows[0]["level"],
        "retained_target_fraction": rows[0]["retained_target_fraction"],
        "contamination_fraction": rows[0]["contamination_fraction"],
        "tree_count": len(rows),
        "mean_unique_output_point_count": float(np.mean([r["unique_output_point_count"] for r in rows])),
        "modalities": {},
        "fusion_vs_point": {},
        "fusion_vs_image": {},
        "by_class": {},
    }
    baseline_rows = [baseline_by_base[row["base_sample_key"]] for row in rows]
    for modality_index, modality in enumerate(MODALITIES):
        current_correct = np.asarray([row[f"{modality}_correct"] for row in rows], dtype=np.float64)
        baseline_correct = np.asarray([row[f"{modality}_correct"] for row in baseline_rows], dtype=np.float64)
        current_probability = np.asarray([row[f"{modality}_true_probability"] for row in rows])
        baseline_probability = np.asarray([row[f"{modality}_true_probability"] for row in baseline_rows])
        baseline_only = int(np.sum((baseline_correct == 1) & (current_correct == 0)))
        current_only = int(np.sum((baseline_correct == 0) & (current_correct == 1)))
        accuracy_delta = current_correct - baseline_correct
        probability_delta = current_probability - baseline_probability
        result["modalities"][modality] = {
            "correct_count": int(current_correct.sum()),
            "accuracy": float(current_correct.mean()),
            "macro_f1": macro_f1(rows, modality),
            "mean_true_probability": float(current_probability.mean()),
            "mean_confidence": float(np.mean([row[f"{modality}_confidence"] for row in rows])),
            "versus_baseline_accuracy_delta": float(accuracy_delta.mean()),
            "versus_baseline_accuracy_delta_bootstrap_95_ci": bootstrap_ci(
                accuracy_delta, 20260819 + seed_offset * 10 + modality_index
            ),
            "baseline_correct_condition_wrong": baseline_only,
            "baseline_wrong_condition_correct": current_only,
            "versus_baseline_mcnemar_exact_two_sided_p": exact_mcnemar_p(baseline_only, current_only),
            "versus_baseline_true_probability_delta": float(probability_delta.mean()),
            "versus_baseline_true_probability_delta_bootstrap_95_ci": bootstrap_ci(
                probability_delta, 202608190 + seed_offset * 10 + modality_index
            ),
        }
    for comparison_name, first, second in (
        ("fusion_vs_point", "fusion", "point"),
        ("fusion_vs_image", "fusion", "image"),
    ):
        first_only = sum(row[f"{first}_correct"] == 1 and row[f"{second}_correct"] == 0 for row in rows)
        second_only = sum(row[f"{first}_correct"] == 0 and row[f"{second}_correct"] == 1 for row in rows)
        result[comparison_name] = {
            f"{first}_correct_{second}_wrong": first_only,
            f"{first}_wrong_{second}_correct": second_only,
            f"{first}_net_correct_gain": first_only - second_only,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(first_only, second_only),
        }
    for class_name in CLASS_NAMES:
        class_rows = [row for row in rows if row["true_species"] == class_name]
        result["by_class"][class_name] = {
            modality: {
                "correct_count": sum(row[f"{modality}_correct"] for row in class_rows),
                "accuracy": float(np.mean([row[f"{modality}_correct"] for row in class_rows])),
                "mean_true_probability": float(np.mean([row[f"{modality}_true_probability"] for row in class_rows])),
            }
            for modality in MODALITIES
        }
    if rows[0]["factor"] == "impurity":
        result["impurity_mechanism"] = {
            "point_prediction_equals_donor_class_count": sum(
                row["point_prediction"] == row["donor_model_class_name"] for row in rows
            ),
            "fusion_prediction_equals_donor_class_count": sum(
                row["fusion_prediction"] == row["donor_model_class_name"] for row in rows
            ),
            "same_road_donor_count": sum(row["donor_same_road"] for row in rows),
        }
    return result


def matched_structure_density(
    rows: list[dict[str, Any]], modality: str, keep: int, seed_offset: int
) -> dict[str, Any]:
    completeness = {
        row["base_sample_key"]: row
        for row in rows
        if row["condition"] == f"completeness_keep{keep}"
    }
    density = {
        row["base_sample_key"]: row
        for row in rows
        if row["condition"] == f"density_keep{keep}"
    }
    keys = sorted(completeness)
    spatial_correct = np.asarray([completeness[key][f"{modality}_correct"] for key in keys], dtype=float)
    random_correct = np.asarray([density[key][f"{modality}_correct"] for key in keys], dtype=float)
    spatial_prob = np.asarray([completeness[key][f"{modality}_true_probability"] for key in keys])
    random_prob = np.asarray([density[key][f"{modality}_true_probability"] for key in keys])
    density_only = int(np.sum((random_correct == 1) & (spatial_correct == 0)))
    spatial_only = int(np.sum((random_correct == 0) & (spatial_correct == 1)))
    accuracy_delta = spatial_correct - random_correct
    probability_delta = spatial_prob - random_prob
    return {
        "keep_percent": keep,
        "modality": modality,
        "pair_count": len(keys),
        "completeness_accuracy": float(spatial_correct.mean()),
        "density_accuracy": float(random_correct.mean()),
        "completeness_minus_density_accuracy": float(accuracy_delta.mean()),
        "accuracy_delta_bootstrap_95_ci": bootstrap_ci(accuracy_delta, 2026081900 + seed_offset),
        "density_correct_completeness_wrong": density_only,
        "density_wrong_completeness_correct": spatial_only,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(density_only, spatial_only),
        "completeness_minus_density_true_probability": float(probability_delta.mean()),
        "true_probability_delta_bootstrap_95_ci": bootstrap_ci(
            probability_delta, 20260819000 + seed_offset
        ),
        "unique_counts_exactly_matched": all(
            completeness[key]["unique_output_point_count"] == density[key]["unique_output_point_count"]
            for key in keys
        ),
    }


def plot_overview(summaries: list[dict[str, Any]], output: Path) -> None:
    labels = [summary["condition"].replace("_", "\n") for summary in summaries]
    x = np.arange(len(summaries))
    width = 0.35
    point = [summary["modalities"]["point"]["accuracy"] * 100 for summary in summaries]
    fusion = [summary["modalities"]["fusion"]["accuracy"] * 100 for summary in summaries]
    image = summaries[0]["modalities"]["image"]["accuracy"] * 100
    fig, axis = plt.subplots(figsize=(14.2, 5.8))
    bars1 = axis.bar(x - width / 2, point, width, label="Point", color=COLORS["point"])
    bars2 = axis.bar(x + width / 2, fusion, width, label="Fusion", color=COLORS["fusion"])
    axis.axhline(image, color=COLORS["image"], linestyle="--", linewidth=2, label=f"Fixed image ({image:.1f}%)")
    axis.bar_label(bars1, fmt="%.1f", padding=2, fontsize=7)
    axis.bar_label(bars2, fmt="%.1f", padding=2, fontsize=7)
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("D2e point-segmentation-quality ablation (n=32)")
    axis.legend(ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_matched(summaries_by_condition: dict[str, dict[str, Any]], output: Path) -> None:
    keep = [100, 75, 50, 25]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for axis, modality in zip(axes, ("point", "fusion")):
        baseline = summaries_by_condition["baseline_full"]["modalities"][modality]["accuracy"] * 100
        completeness = [baseline] + [
            summaries_by_condition[f"completeness_keep{value}"]["modalities"][modality]["accuracy"] * 100
            for value in keep[1:]
        ]
        density = [baseline] + [
            summaries_by_condition[f"density_keep{value}"]["modalities"][modality]["accuracy"] * 100
            for value in keep[1:]
        ]
        axis.plot(keep, completeness, marker="o", linewidth=2.2, label="Spatial completeness", color="#D55E00")
        axis.plot(keep, density, marker="o", linewidth=2.2, label="Random density control", color="#56B4E9")
        axis.invert_xaxis()
        axis.set_xticks(keep)
        axis.set_ylim(0, 105)
        axis.set_xlabel("Retained unique target points (%)")
        axis.set_title(modality.title())
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Spatial loss versus matched-count random thinning")
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_impurity(summaries_by_condition: dict[str, dict[str, Any]], output: Path) -> None:
    contamination = [0, 10, 25, 40]
    names = ["baseline_full", "impurity10", "impurity25", "impurity40"]
    fig, axis = plt.subplots(figsize=(8.8, 5.0))
    for modality in ("point", "fusion"):
        values = [summaries_by_condition[name]["modalities"][modality]["accuracy"] * 100 for name in names]
        axis.plot(contamination, values, marker="o", linewidth=2.2, label=modality.title(), color=COLORS[modality])
        for x, y in zip(contamination, values):
            axis.annotate(f"{y:.1f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    image = summaries_by_condition["baseline_full"]["modalities"]["image"]["accuracy"] * 100
    axis.axhline(image, color=COLORS["image"], linestyle="--", linewidth=2, label=f"Fixed image ({image:.1f}%)")
    axis.set_xticks(contamination)
    axis.set_ylim(0, 105)
    axis.set_xlabel("Cross-class contamination (%)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Segmentation impurity dose-response")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_class_heatmap(summaries: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.9), sharey=True)
    labels = [summary["condition"].replace("completeness_", "C-").replace("density_", "D-").replace("baseline_full", "Base").replace("impurity", "I-") for summary in summaries]
    for axis, modality in zip(axes, ("point", "fusion")):
        matrix = np.asarray(
            [
                [summary["by_class"][class_name][modality]["accuracy"] * 100 for summary in summaries]
                for class_name in CLASS_NAMES
            ]
        )
        axis.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value >= 62.5 else "black",
                )
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_title(f"{modality.title()} accuracy (%)")
    fig.suptitle("Class-specific response to point segmentation quality")
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.25, top=0.82, wspace=0.08)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = merge(root)
    condition_names = [str(item["name"]) for item in manifest["quality_conditions"]]
    by_condition = {
        name: [row for row in rows if row["condition"] == name] for name in condition_names
    }
    baseline_by_base = {
        row["base_sample_key"]: row for row in by_condition["baseline_full"]
    }
    summaries = [
        condition_summary(by_condition[name], baseline_by_base, index)
        for index, name in enumerate(condition_names)
    ]
    summaries_by_condition = {summary["condition"]: summary for summary in summaries}

    nonbaseline = [summary for summary in summaries if summary["condition"] != "baseline_full"]
    holm_families: dict[str, dict[str, float]] = {}
    for modality in ("point", "fusion"):
        adjusted = holm_adjust(
            {
                summary["condition"]: summary["modalities"][modality]["versus_baseline_mcnemar_exact_two_sided_p"]
                for summary in nonbaseline
            }
        )
        holm_families[f"{modality}_versus_baseline"] = adjusted
        for summary in nonbaseline:
            summary["modalities"][modality]["versus_baseline_holm_adjusted_p"] = adjusted[summary["condition"]]

    matched = [
        matched_structure_density(rows, modality, keep, modality_index * 10 + keep)
        for modality_index, modality in enumerate(("point", "fusion"))
        for keep in (75, 50, 25)
    ]
    for modality in ("point", "fusion"):
        subset = [item for item in matched if item["modality"] == modality]
        adjusted = holm_adjust({str(item["keep_percent"]): item["mcnemar_exact_two_sided_p"] for item in subset})
        holm_families[f"{modality}_completeness_vs_density"] = adjusted
        for item in subset:
            item["holm_adjusted_p"] = adjusted[str(item["keep_percent"])]

    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[row["base_sample_key"]].append(row)
    image_prediction_invariant = all(
        len({row["image_prediction"] for row in base_rows}) == 1 for base_rows in by_base.values()
    )
    max_image_probability_spread = max(
        max(row["image_true_probability"] for row in base_rows)
        - min(row["image_true_probability"] for row in base_rows)
        for base_rows in by_base.values()
    )
    result = {
        "format_version": 1,
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "design": manifest["design"],
        "interpretation_limit": manifest["interpretation_limit"],
        "base_tree_count": len(by_base),
        "record_count": len(rows),
        "class_counts": dict(Counter(row["true_species"] for row in by_condition["baseline_full"])),
        "image_prediction_invariant_across_point_conditions": image_prediction_invariant,
        "max_image_true_probability_spread": max_image_probability_spread,
        "multiple_comparison_adjustment": {
            "method": "Holm family-wise correction within predefined comparison families",
            "families": holm_families,
        },
        "condition_summaries": summaries,
        "matched_completeness_density": matched,
    }
    summary_path = output_dir / "d2e_point_quality_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "d2e_per_variant.csv", rows)
    flat = []
    for summary in summaries:
        for modality, metrics in summary["modalities"].items():
            flat.append(
                {
                    "condition": summary["condition"],
                    "factor": summary["factor"],
                    "level": summary["level"],
                    "retained_target_fraction": summary["retained_target_fraction"],
                    "contamination_fraction": summary["contamination_fraction"],
                    "mean_unique_output_point_count": summary["mean_unique_output_point_count"],
                    "modality": modality,
                    **metrics,
                }
            )
    write_csv(output_dir / "d2e_condition_modality_summary.csv", flat)
    write_csv(output_dir / "d2e_matched_completeness_density.csv", matched)
    plot_overview(summaries, output_dir / "d2e_accuracy_overview.png")
    plot_matched(summaries_by_condition, output_dir / "d2e_completeness_vs_density.png")
    plot_impurity(summaries_by_condition, output_dir / "d2e_impurity_dose_response.png")
    plot_class_heatmap(summaries, output_dir / "d2e_class_heatmap.png")

    validation = {
        "status": "passed",
        "generated_at": result["generated_at"],
        "base_tree_count": len(by_base),
        "condition_count": len(condition_names),
        "record_count": len(rows),
        "expected_record_count": len(by_base) * len(condition_names),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "all_conditions_balanced": len({len(value) for value in by_condition.values()}) == 1,
        "image_prediction_invariant": image_prediction_invariant,
        "max_image_true_probability_spread": max_image_probability_spread,
        "all_matched_unique_counts_equal": all(item["unique_counts_exactly_matched"] for item in matched),
        "summary_sha256": sha256(summary_path),
    }
    if not (
        validation["record_count"] == validation["expected_record_count"]
        and validation["all_conditions_balanced"]
        and validation["image_prediction_invariant"]
        and validation["all_matched_unique_counts_equal"]
    ):
        validation["status"] = "failed"
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
