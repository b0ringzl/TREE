"""Create a paired NJ test comparison between YOLO11s-cls and WHU ResNet50."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-run-dir", type=Path, required=True)
    parser.add_argument("--resnet-suite", type=Path, required=True)
    parser.add_argument("--resnet-seed", type=int, default=20260728)
    return parser.parse_args()


def load_predictions(path: Path, split: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["sample_key"]: row for row in csv.DictReader(stream) if row["split"] == split}


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    lower = min(successes, trials - successes)
    probability = sum(math.comb(trials, value) for value in range(lower + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def main() -> None:
    args = parse_args()
    yolo_run = args.yolo_run_dir.resolve()
    resnet_run = args.resnet_suite.resolve() / "runs" / f"seed_{args.resnet_seed}"
    yolo_metrics = json.loads((yolo_run / "final_metrics.json").read_text(encoding="utf-8"))
    yolo_environment = json.loads((yolo_run / "environment.json").read_text(encoding="utf-8"))
    yolo_seed = int(yolo_environment["seed"])
    if yolo_seed != args.resnet_seed:
        raise ValueError(
            f"Seed mismatch: YOLO seed={yolo_seed}, requested ResNet seed={args.resnet_seed}"
        )
    resnet_metrics = json.loads((resnet_run / "final_metrics.json").read_text(encoding="utf-8"))
    resnet_seed_metrics = []
    for candidate in sorted((args.resnet_suite.resolve() / "runs").glob("seed_*")):
        metrics_path = candidate / "final_metrics.json"
        if metrics_path.is_file():
            candidate_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]["test"]
            resnet_seed_metrics.append(
                {
                    "seed": int(candidate.name.removeprefix("seed_")),
                    "accuracy": float(candidate_metrics["accuracy"]),
                    "macro_f1": float(candidate_metrics["macro_f1"]),
                }
            )
    yolo_predictions = load_predictions(yolo_run / "predictions.csv", "test")
    resnet_predictions = load_predictions(resnet_run / "predictions.csv", "test")
    common = sorted(set(yolo_predictions) & set(resnet_predictions))
    if len(common) != 151 or set(yolo_predictions) != set(resnet_predictions):
        raise ValueError(
            f"Paired test mismatch: YOLO={len(yolo_predictions)}, ResNet={len(resnet_predictions)}, common={len(common)}"
        )
    for sample_key in common:
        if yolo_predictions[sample_key]["true_class"] != resnet_predictions[sample_key]["true_class"]:
            raise ValueError(f"Ground-truth mismatch: {sample_key}")
    yolo_correct = np.array([int(yolo_predictions[key]["correct"]) for key in common], dtype=np.int8)
    resnet_correct = np.array([int(resnet_predictions[key]["correct"]) for key in common], dtype=np.int8)
    resnet_only = int(np.sum((resnet_correct == 1) & (yolo_correct == 0)))
    yolo_only = int(np.sum((resnet_correct == 0) & (yolo_correct == 1)))
    both_correct = int(np.sum((resnet_correct == 1) & (yolo_correct == 1)))
    both_wrong = int(np.sum((resnet_correct == 0) & (yolo_correct == 0)))
    rng = np.random.default_rng(20260824)
    differences = np.empty(10000, dtype=np.float64)
    for index in range(len(differences)):
        sampled = rng.integers(0, len(common), size=len(common))
        differences[index] = yolo_correct[sampled].mean() - resnet_correct[sampled].mean()
    yolo_test = yolo_metrics["metrics"]["test"]
    resnet_test = resnet_metrics["metrics"]["test"]
    payload = {
        "status": "complete",
        "scope": "same NJ single-tree crops, same frozen split, same four classes, paired test samples",
        "class_names": yolo_metrics["class_names"],
        "test_samples": len(common),
        "yolo11s_cls": {
            "seed": yolo_seed,
            "accuracy": yolo_test["accuracy"],
            "balanced_accuracy": yolo_test["balanced_accuracy"],
            "macro_f1": yolo_test["macro_f1"],
            "checkpoint": yolo_metrics["checkpoint"],
        },
        "whu_resnet50": {
            "seed": args.resnet_seed,
            "accuracy": resnet_test["accuracy"],
            "balanced_accuracy": resnet_test["balanced_accuracy"],
            "macro_f1": resnet_test["macro_f1"],
            "checkpoint": str(resnet_run / "best.pt"),
        },
        "whu_resnet50_three_seed_summary": {
            "runs": resnet_seed_metrics,
            "accuracy_mean": statistics.mean(row["accuracy"] for row in resnet_seed_metrics),
            "accuracy_sample_std": statistics.stdev(row["accuracy"] for row in resnet_seed_metrics),
            "macro_f1_mean": statistics.mean(row["macro_f1"] for row in resnet_seed_metrics),
            "macro_f1_sample_std": statistics.stdev(row["macro_f1"] for row in resnet_seed_metrics),
        },
        "paired_accuracy_difference_yolo_minus_resnet": float(yolo_correct.mean() - resnet_correct.mean()),
        "paired_bootstrap_95ci": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
        "paired_outcomes": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "resnet_only_correct": resnet_only,
            "yolo_only_correct": yolo_only,
        },
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(
            min(resnet_only, yolo_only), resnet_only + yolo_only
        ),
        "limitations": [
            "This file is a paired single-seed comparison; the three-seed aggregate is reported separately.",
            "The recipes are model-native rather than identical: YOLO11 uses AdamW, ResNet50 uses SGD.",
            "This compares classification on pre-cropped target trees, not full-image detection performance.",
            "Trees from the same road or trajectory may be correlated; the sample-level McNemar p-value is exploratory.",
        ],
    }
    (yolo_run / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metric_labels = ["Accuracy", "Balanced accuracy", "Macro-F1"]
    yolo_values = [
        float(yolo_test["accuracy"]),
        float(yolo_test["balanced_accuracy"]),
        float(yolo_test["macro_f1"]),
    ]
    resnet_values = [
        float(resnet_test["accuracy"]),
        float(resnet_test["balanced_accuracy"]),
        float(resnet_test["macro_f1"]),
    ]
    yolo_per_class = [float(row["f1"]) for row in yolo_test["per_class"]]
    resnet_per_class = [float(row["f1"]) for row in resnet_test["per_class"]]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    width = 0.36
    x_metrics = np.arange(len(metric_labels))
    axes[0].bar(x_metrics - width / 2, yolo_values, width, label="YOLO11s-cls")
    axes[0].bar(x_metrics + width / 2, resnet_values, width, label="WHU ResNet50")
    axes[0].set_xticks(x_metrics, metric_labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("NJ paired test metrics (n=151)")
    axes[0].legend()
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=2, fontsize=9)
    short_names = ["C. camphora", "L. indica", "M. grandiflora", "Other"]
    x_classes = np.arange(len(short_names))
    axes[1].bar(x_classes - width / 2, yolo_per_class, width, label="YOLO11s-cls")
    axes[1].bar(x_classes + width / 2, resnet_per_class, width, label="WHU ResNet50")
    axes[1].set_xticks(x_classes, short_names, rotation=20, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Per-class F1")
    axes[1].legend()
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    figure.savefig(yolo_run / "comparison_metrics.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
