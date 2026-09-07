"""Analyze frozen D2 image, point, and fusion ensemble predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_frozen_three_modality_evaluation_v1"
)
SUITE_ROOTS = {
    "image": DERIVED_ROOT / "d1_resnet50_clean_repeats" / "20260817_protocol_v1",
    "point": DERIVED_ROOT / "d1_ptv2_clean_repeats" / "20260818_protocol_v1",
    "fusion": DERIVED_ROOT / "d1_fusion_clean_repeats" / "20260818_protocol_v1",
}
SEEDS = (20260728, 20260729, 20260730)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def classification_summary(rows: list[dict[str, Any]], modality: str, class_names: list[str]) -> dict[str, Any]:
    correct_field = f"{modality}_correct"
    prediction_field = f"{modality}_prediction"
    correct = sum(int(row[correct_field]) for row in rows)
    per_class = []
    for name in class_names:
        tp = sum(row["true_species"] == name and row[prediction_field] == name for row in rows)
        fp = sum(row["true_species"] != name and row[prediction_field] == name for row in rows)
        fn = sum(row["true_species"] == name and row[prediction_field] != name for row in rows)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_name": name,
                "support": support,
                "correct": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "sample_count": len(rows),
        "correct_count": correct,
        "accuracy": correct / len(rows),
        "accuracy_wilson_95": wilson(correct, len(rows)),
        "macro_precision": statistics.fmean(item["precision"] for item in per_class),
        "macro_recall": statistics.fmean(item["recall"] for item in per_class),
        "macro_f1": statistics.fmean(item["f1"] for item in per_class),
        "per_class": per_class,
    }


def exact_paired_p_value(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) * 0.5**discordant
        for value in range(0, min(first_only, second_only) + 1)
    )
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    comparison_path = root / "fusion" / "three_modality_comparison.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8-sig"))
    records = {str(item["sample_key"]): item for item in manifest["records"]}
    rows: list[dict[str, Any]] = []
    for item in comparison["rows"]:
        record = records[str(item["sample_key"])]
        rows.append(
            {
                "sample_key": str(item["sample_key"]),
                "tree_id": int(item["tree_id"]),
                "road_id": str(record["road_id"]),
                "trajectory_id": str(record["trajectory_id"]),
                "exposure_group": str(item["exposure_group"]),
                "true_species": str(item["true_species"]),
                "source_scientific_name": str(record["source_scientific_name"]),
                "target_mean_luminance": float(record["d2_target_metrics"]["target_r5_mean_luminance"]),
                "target_edge_energy": float(record["d2_target_metrics"]["target_r5_edge_energy"]),
                "image_prediction": str(item["image_prediction"]),
                "image_correct": int(item["image_correct"]),
                "image_confidence": float(item["image_confidence"]),
                "point_prediction": str(item["point_prediction"]),
                "point_correct": int(item["point_correct"]),
                "point_confidence": float(item["point_confidence"]),
                "fusion_prediction": str(item["fusion_prediction"]),
                "fusion_correct": int(item["fusion_correct"]),
                "fusion_confidence": float(item["fusion_confidence"]),
                "fusion_vs_image": str(item["fusion_vs_image"]),
                "fusion_vs_point": str(item["fusion_vs_point"]),
            }
        )
    if set(records) != {row["sample_key"] for row in rows}:
        raise ValueError("Manifest and comparison cohorts differ")
    class_names = [
        str(item["scientific_name"])
        for item in sorted(manifest["classes"], key=lambda item: int(item["class_index"]))
    ]
    modalities = ("image", "point", "fusion")
    overall = {name: classification_summary(rows, name, class_names) for name in modalities}
    by_exposure: dict[str, dict[str, Any]] = {}
    for group in ("bright", "dark"):
        group_rows = [row for row in rows if row["exposure_group"] == group]
        by_exposure[group] = {
            name: classification_summary(group_rows, name, class_names) for name in modalities
        }
    target_rows = [row for row in rows if row["true_species"] != "Other"]
    other_rows = [row for row in rows if row["true_species"] == "Other"]
    cohorts = {
        "three_target_classes": {
            name: classification_summary(target_rows, name, class_names) for name in modalities
        },
        "other": {
            name: classification_summary(other_rows, name, class_names) for name in modalities
        },
    }
    paired: dict[str, Any] = {}
    for first, second in (("image", "point"), ("image", "fusion"), ("point", "fusion")):
        first_only = sum(
            int(row[f"{first}_correct"] and not row[f"{second}_correct"]) for row in rows
        )
        second_only = sum(
            int(row[f"{second}_correct"] and not row[f"{first}_correct"]) for row in rows
        )
        paired[f"{first}_vs_{second}"] = {
            f"{first}_only_correct": first_only,
            f"{second}_only_correct": second_only,
            "net_corrected_by_second": second_only - first_only,
            "exact_mcnemar_p_value": exact_paired_p_value(first_only, second_only),
        }
    stability = {
        name: {
            "bright_accuracy": by_exposure["bright"][name]["accuracy"],
            "dark_accuracy": by_exposure["dark"][name]["accuracy"],
            "absolute_bright_dark_gap": abs(
                by_exposure["bright"][name]["accuracy"]
                - by_exposure["dark"][name]["accuracy"]
            ),
        }
        for name in modalities
    }
    road_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        road_rows[row["road_id"]].append(row)
    by_road = {
        road: {
            name: classification_summary(group, name, class_names) for name in modalities
        }
        for road, group in sorted(road_rows.items())
    }
    clean_reference: dict[str, Any] = {}
    for name, suite_root in SUITE_ROOTS.items():
        seed_values = []
        for seed in SEEDS:
            final_path = suite_root / "runs" / f"seed_{seed}" / "final_metrics.json"
            final = json.loads(final_path.read_text(encoding="utf-8-sig"))
            seed_values.append(
                {
                    "seed": seed,
                    "accuracy": float(final["metrics"]["test"]["accuracy"]),
                    "macro_f1": float(final["metrics"]["test"]["macro_f1"]),
                }
            )
        clean_reference[name] = {
            "per_seed": seed_values,
            "accuracy_mean": statistics.fmean(item["accuracy"] for item in seed_values),
            "macro_f1_mean": statistics.fmean(item["macro_f1"] for item in seed_values),
            "comparison_limit": "D1 clean test is not class/road matched to the D2 exposure cohort.",
        }

    summary = {
        "format_version": 1,
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cohort": manifest["summary"],
        "overall": overall,
        "by_exposure": by_exposure,
        "cohorts": cohorts,
        "paired": paired,
        "exposure_stability": stability,
        "by_road": by_road,
        "clean_test_reference": clean_reference,
        "primary_interpretation": {
            "fusion_vs_image_accuracy_delta": overall["fusion"]["accuracy"] - overall["image"]["accuracy"],
            "point_vs_image_accuracy_delta": overall["point"]["accuracy"] - overall["image"]["accuracy"],
            "dark_point_vs_image_delta": by_exposure["dark"]["point"]["accuracy"] - by_exposure["dark"]["image"]["accuracy"],
            "dark_fusion_vs_image_delta": by_exposure["dark"]["fusion"]["accuracy"] - by_exposure["dark"]["image"]["accuracy"],
            "bright_fusion_vs_image_delta": by_exposure["bright"]["fusion"]["accuracy"] - by_exposure["bright"]["image"]["accuracy"],
            "class_imbalance_warning": "55/63 samples are Other; only 8 belong to the three named target classes.",
            "causal_claim_status": "not established; matched normal controls are still required",
        },
    }
    summary_path = root / "d2_three_modality_summary.json"
    rows_path = root / "d2_per_tree_modalities.csv"
    atomic_json(summary_path, summary)
    atomic_csv(rows_path, rows)

    cohort_names = ["overall", "bright", "dark", "three target", "Other"]
    values = {
        name: [
            overall[name]["accuracy"],
            by_exposure["bright"][name]["accuracy"],
            by_exposure["dark"][name]["accuracy"],
            cohorts["three_target_classes"][name]["accuracy"],
            cohorts["other"][name]["accuracy"],
        ]
        for name in modalities
    }
    plt.figure(figsize=(11, 6.2))
    x = list(range(len(cohort_names)))
    width = 0.24
    colors = {"image": "#4472C4", "point": "#70AD47", "fusion": "#ED7D31"}
    labels = {"image": "Image", "point": "Point cloud", "fusion": "Fusion"}
    for index, name in enumerate(modalities):
        positions = [value + (index - 1) * width for value in x]
        bars = plt.bar(positions, values[name], width, label=labels[name], color=colors[name])
        for bar, value in zip(bars, values[name], strict=True):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.xticks(x, ["Overall\nn=63", "Bright\nn=53", "Dark\nn=10", "3 target classes\nn=8", "Other\nn=55"])
    plt.ylim(0, 1.08)
    plt.ylabel("Three-seed probability-ensemble accuracy")
    plt.title("D2 user-confirmed exposure cohort: three-modality comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=3, loc="upper center")
    plt.tight_layout()
    plot_path = root / "d2_three_modality_accuracy.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    validation = {
        "status": "passed",
        "generated_at": summary["generated_at"],
        "sample_count": len(rows),
        "keys_unique": len({row["sample_key"] for row in rows}) == len(rows),
        "modalities_complete": all(
            f"{name}_prediction" in row and f"{name}_correct" in row
            for row in rows
            for name in modalities
        ),
        "inputs": {
            "manifest": sha256_file(manifest_path),
            "comparison": sha256_file(comparison_path),
        },
        "outputs": {
            summary_path.name: sha256_file(summary_path),
            rows_path.name: sha256_file(rows_path),
            plot_path.name: sha256_file(plot_path),
        },
    }
    atomic_json(root / "d2_analysis_validation.json", validation)
    print(
        json.dumps(
            {
                "status": "passed",
                "overall_accuracy": {name: overall[name]["accuracy"] for name in modalities},
                "macro_f1": {name: overall[name]["macro_f1"] for name in modalities},
                "by_exposure_accuracy": {
                    group: {name: by_exposure[group][name]["accuracy"] for name in modalities}
                    for group in ("bright", "dark")
                },
                "paired": paired,
                "plot": str(plot_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
