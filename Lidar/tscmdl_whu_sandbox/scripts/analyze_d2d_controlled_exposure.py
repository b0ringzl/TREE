#!/usr/bin/env python3
"""Analyze the balanced same-tree controlled exposure experiment."""

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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
    if vector.size == 1:
        return [float(vector[0]), float(vector[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 1000):
        stop = min(start + 1000, repetitions)
        indices = rng.integers(0, vector.size, size=(stop - start, vector.size))
        means[start:stop] = vector[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def macro_f1(rows: list[dict[str, Any]], modality: str, class_names: list[str]) -> float:
    values = []
    for class_name in class_names:
        tp = sum(r["true_species"] == class_name and r[f"{modality}_prediction"] == class_name for r in rows)
        fp = sum(r["true_species"] != class_name and r[f"{modality}_prediction"] == class_name for r in rows)
        fn = sum(r["true_species"] == class_name and r[f"{modality}_prediction"] != class_name for r in rows)
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(values))


def merge(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_json(root / "manifest.json")
    image = load_csv(root / "image" / "per_tree_summary.csv")
    point = load_csv(root / "point" / "per_tree_summary.csv")
    fusion_doc = load_json(root / "fusion" / "three_modality_comparison.json")
    fusion = {str(row["sample_key"]): row for row in fusion_doc["rows"]}
    target_doc = load_json(root / "target_metrics" / "target_exposure_metrics.json")
    target = {str(row["sample_key"]): row for row in target_doc["rows"]}
    manifest_by_key = {str(row["sample_key"]): row for row in manifest["records"]}
    rows: list[dict[str, Any]] = []
    for key, record in manifest_by_key.items():
        image_row = image[key]
        point_row = point[key]
        fusion_row = fusion[key]
        target_row = target[key]
        row: dict[str, Any] = {
            "sample_key": key,
            "base_sample_key": str(record["base_sample_key"]),
            "tree_id": int(record["tree_id"]),
            "road_id": str(record["road_id"]),
            "source_scientific_name": str(record["source_scientific_name"]),
            "true_species": str(record["scientific_name"]),
            "class_index": int(record["class_index"]),
            "condition": str(record["exposure_condition"]),
            "condition_order": int(record["exposure_condition_order"]),
            "ev_shift": float(record["exposure_ev_shift"]),
            "direction": str(record["exposure_direction"]),
            "severity": int(record["exposure_severity"]),
            "target_mean_luminance": float(target_row["target_r5_mean_luminance"]),
            "target_dark_ratio": float(target_row["target_r5_dark_ratio"]),
            "target_bright_ratio": float(target_row["target_r5_bright_ratio"]),
            "target_edge_energy": float(target_row["target_r5_edge_energy"]),
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
    return rows, manifest, manifest_by_key


def condition_summary(
    condition_rows: list[dict[str, Any]],
    normal_by_base: dict[str, dict[str, Any]],
    class_names: list[str],
    seed_offset: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "condition": condition_rows[0]["condition"],
        "ev_shift": condition_rows[0]["ev_shift"],
        "direction": condition_rows[0]["direction"],
        "severity": condition_rows[0]["severity"],
        "tree_count": len(condition_rows),
        "target_metrics": {
            "mean_luminance": float(np.mean([r["target_mean_luminance"] for r in condition_rows])),
            "dark_ratio": float(np.mean([r["target_dark_ratio"] for r in condition_rows])),
            "bright_ratio": float(np.mean([r["target_bright_ratio"] for r in condition_rows])),
            "edge_energy": float(np.mean([r["target_edge_energy"] for r in condition_rows])),
        },
        "modalities": {},
        "fusion_vs_image": {},
        "point_vs_image": {},
        "by_class": {},
    }
    for modality_index, modality in enumerate(MODALITIES):
        current_correct = np.asarray([r[f"{modality}_correct"] for r in condition_rows], dtype=np.float64)
        current_probability = np.asarray([r[f"{modality}_true_probability"] for r in condition_rows])
        normal_rows = [normal_by_base[r["base_sample_key"]] for r in condition_rows]
        normal_correct = np.asarray([r[f"{modality}_correct"] for r in normal_rows], dtype=np.float64)
        normal_probability = np.asarray([r[f"{modality}_true_probability"] for r in normal_rows])
        normal_only = int(np.sum((normal_correct == 1) & (current_correct == 0)))
        current_only = int(np.sum((normal_correct == 0) & (current_correct == 1)))
        accuracy_delta = current_correct - normal_correct
        probability_delta = current_probability - normal_probability
        result["modalities"][modality] = {
            "correct_count": int(current_correct.sum()),
            "accuracy": float(current_correct.mean()),
            "macro_f1": macro_f1(condition_rows, modality, class_names),
            "mean_true_probability": float(current_probability.mean()),
            "mean_confidence": float(np.mean([r[f"{modality}_confidence"] for r in condition_rows])),
            "versus_normal_accuracy_delta": float(accuracy_delta.mean()),
            "versus_normal_accuracy_delta_bootstrap_95_ci": bootstrap_ci(
                accuracy_delta, 202608190 + seed_offset * 10 + modality_index
            ),
            "normal_correct_condition_wrong": normal_only,
            "normal_wrong_condition_correct": current_only,
            "versus_normal_mcnemar_exact_two_sided_p": exact_mcnemar_p(normal_only, current_only),
            "versus_normal_true_probability_delta": float(probability_delta.mean()),
            "versus_normal_true_probability_delta_bootstrap_95_ci": bootstrap_ci(
                probability_delta, 2026081900 + seed_offset * 10 + modality_index
            ),
        }
    for comparison_name, first, second in (
        ("fusion_vs_image", "fusion", "image"),
        ("point_vs_image", "point", "image"),
    ):
        first_only = sum(r[f"{first}_correct"] == 1 and r[f"{second}_correct"] == 0 for r in condition_rows)
        second_only = sum(r[f"{first}_correct"] == 0 and r[f"{second}_correct"] == 1 for r in condition_rows)
        second_error_count = sum(r[f"{second}_correct"] == 0 for r in condition_rows)
        result[comparison_name] = {
            f"{first}_correct_{second}_wrong": first_only,
            f"{first}_wrong_{second}_correct": second_only,
            f"{first}_net_correct_gain": first_only - second_only,
            f"{first}_recovery_rate_among_{second}_errors": (
                first_only / second_error_count if second_error_count else 0.0
            ),
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(first_only, second_only),
        }
    for class_name in class_names:
        class_rows = [row for row in condition_rows if row["true_species"] == class_name]
        result["by_class"][class_name] = {
            modality: {
                "correct_count": sum(r[f"{modality}_correct"] for r in class_rows),
                "accuracy": float(np.mean([r[f"{modality}_correct"] for r in class_rows])),
                "mean_true_probability": float(np.mean([r[f"{modality}_true_probability"] for r in class_rows])),
            }
            for modality in MODALITIES
        }
    return result


def plot_accuracy(summaries: list[dict[str, Any]], output: Path) -> None:
    ev = [summary["ev_shift"] for summary in summaries]
    fig, axis = plt.subplots(figsize=(9.4, 5.3))
    for modality in MODALITIES:
        accuracy = [summary["modalities"][modality]["accuracy"] * 100 for summary in summaries]
        axis.plot(ev, accuracy, marker="o", linewidth=2.2, label=modality.title(), color=COLORS[modality])
        vertical_offset = {"image": -15, "point": 8, "fusion": 8}[modality]
        for x, y in zip(ev, accuracy):
            axis.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, vertical_offset),
                ha="center",
                fontsize=8,
            )
    axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
    axis.set_xticks(ev, [f"{value:+.0f} EV" if value else "Normal" for value in ev])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("D2d same-tree controlled exposure: accuracy dose-response (n=32)")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_probability(summaries: list[dict[str, Any]], output: Path) -> None:
    ev = [summary["ev_shift"] for summary in summaries]
    fig, axis = plt.subplots(figsize=(9.4, 5.3))
    for modality in MODALITIES:
        probability = [summary["modalities"][modality]["mean_true_probability"] for summary in summaries]
        axis.plot(ev, probability, marker="o", linewidth=2.2, label=modality.title(), color=COLORS[modality])
    axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
    axis.set_xticks(ev, [f"{value:+.0f} EV" if value else "Normal" for value in ev])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Mean true-class probability")
    axis.set_title("True-class evidence under controlled exposure")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_recovery(summaries: list[dict[str, Any]], output: Path) -> None:
    labels = [summary["condition"] for summary in summaries if summary["condition"] != "normal"]
    values = [
        summary["fusion_vs_image"]["fusion_net_correct_gain"]
        for summary in summaries
        if summary["condition"] != "normal"
    ]
    colors = ["#56B4E9" if value >= 0 else "#D55E00" for value in values]
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    bars = axis.bar(labels, values, color=colors)
    axis.bar_label(bars, fmt="%+d", padding=3)
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_ylabel("Fusion net correct gain over image (trees)")
    axis.set_title("Does fusion recover image errors at each exposure level?")
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_class_heatmap(summaries: list[dict[str, Any]], class_names: list[str], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    for axis, modality in zip(axes, ("image", "fusion")):
        matrix = np.asarray(
            [
                [summary["by_class"][class_name][modality]["accuracy"] * 100 for summary in summaries]
                for class_name in class_names
            ]
        )
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[row, column] >= 62.5 else "black",
                )
        axis.set_xticks(range(len(summaries)), [f"{s['ev_shift']:+.0f}" for s in summaries])
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Exposure shift (EV)")
        axis.set_title(f"{modality.title()} accuracy (%)")
    fig.suptitle("Class-specific controlled exposure response (8 trees per class)")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.84, wspace=0.1)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest, manifest_by_key = merge(root)
    class_names = [
        str(item["scientific_name"])
        for item in sorted(manifest["classes"], key=lambda item: int(item["class_index"]))
    ]
    conditions = [str(item["name"]) for item in manifest["exposure_conditions"]]
    by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in conditions}
    normal_by_base = {row["base_sample_key"]: row for row in by_condition["normal"]}
    summaries = [
        condition_summary(by_condition[condition], normal_by_base, class_names, index)
        for index, condition in enumerate(conditions)
    ]
    nonnormal = [summary for summary in summaries if summary["condition"] != "normal"]
    holm_families: dict[str, dict[str, float]] = {}
    for modality in MODALITIES:
        family_name = f"{modality}_versus_normal"
        adjusted = holm_adjust(
            {
                summary["condition"]: summary["modalities"][modality]["versus_normal_mcnemar_exact_two_sided_p"]
                for summary in nonnormal
            }
        )
        holm_families[family_name] = adjusted
        for summary in nonnormal:
            summary["modalities"][modality]["versus_normal_holm_adjusted_p"] = adjusted[summary["condition"]]
    for comparison_name in ("fusion_vs_image", "point_vs_image"):
        adjusted = holm_adjust(
            {
                summary["condition"]: summary[comparison_name]["mcnemar_exact_two_sided_p"]
                for summary in nonnormal
            }
        )
        holm_families[comparison_name] = adjusted
        for summary in nonnormal:
            summary[comparison_name]["holm_adjusted_p"] = adjusted[summary["condition"]]

    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[row["base_sample_key"]].append(row)
    point_prediction_invariant = all(
        len({row["point_prediction"] for row in base_rows}) == 1 for base_rows in by_base.values()
    )
    max_point_probability_spread = max(
        max(row["point_true_probability"] for row in base_rows)
        - min(row["point_true_probability"] for row in base_rows)
        for base_rows in by_base.values()
    )
    target_luminance_monotonic = all(
        all(
            next(row for row in base_rows if row["ev_shift"] == left)["target_mean_luminance"]
            < next(row for row in base_rows if row["ev_shift"] == right)["target_mean_luminance"]
            for left, right in zip([-3, -2, -1, 0, 1, 2], [-2, -1, 0, 1, 2, 3])
        )
        for base_rows in by_base.values()
    )
    result = {
        "format_version": 1,
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "design": manifest["design"],
        "interpretation_limit": manifest["interpretation_limit"],
        "base_tree_count": manifest["summary"]["base_tree_count"],
        "record_count": len(rows),
        "class_counts": dict(Counter(row["true_species"] for row in by_condition["normal"])),
        "point_prediction_invariant_across_exposure": point_prediction_invariant,
        "max_point_true_probability_spread_across_repeated_inputs": max_point_probability_spread,
        "target_luminance_strictly_monotonic_for_every_tree": target_luminance_monotonic,
        "multiple_comparison_adjustment": {
            "method": "Holm family-wise correction across six non-normal exposure levels",
            "families": holm_families,
        },
        "condition_summaries": summaries,
    }
    summary_path = output_dir / "d2d_controlled_exposure_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "d2d_per_variant.csv", rows)

    flat: list[dict[str, Any]] = []
    for summary in summaries:
        for modality, metrics in summary["modalities"].items():
            flat.append(
                {
                    "condition": summary["condition"],
                    "ev_shift": summary["ev_shift"],
                    "target_mean_luminance": summary["target_metrics"]["mean_luminance"],
                    "modality": modality,
                    **metrics,
                }
            )
    write_csv(output_dir / "d2d_condition_modality_summary.csv", flat)
    plot_accuracy(summaries, output_dir / "d2d_accuracy_dose_response.png")
    plot_probability(summaries, output_dir / "d2d_true_probability_dose_response.png")
    plot_recovery(summaries, output_dir / "d2d_fusion_recovery.png")
    plot_class_heatmap(summaries, class_names, output_dir / "d2d_class_heatmap.png")

    validation = {
        "status": "passed",
        "generated_at": result["generated_at"],
        "base_tree_count": len(by_base),
        "condition_count": len(conditions),
        "record_count": len(rows),
        "expected_record_count": len(by_base) * len(conditions),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "all_conditions_balanced": len({len(value) for value in by_condition.values()}) == 1,
        "point_prediction_invariant": point_prediction_invariant,
        "max_point_true_probability_spread": max_point_probability_spread,
        "target_luminance_monotonic": target_luminance_monotonic,
        "all_variant_point_hashes_match_base": all(
            record["packaged_point_sha256"]
            == next(
                base["packaged_point_sha256"]
                for base in manifest["base_records"]
                if base["base_sample_key"] == record["base_sample_key"]
            )
            for record in manifest_by_key.values()
        ),
        "summary_sha256": sha256(summary_path),
    }
    if not (
        validation["record_count"] == validation["expected_record_count"]
        and validation["all_conditions_balanced"]
        and validation["point_prediction_invariant"]
        and validation["target_luminance_monotonic"]
        and validation["all_variant_point_hashes_match_base"]
    ):
        validation["status"] = "failed"
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
