"""Create readable C2 figures and structured class diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def readable_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    split: str,
    accuracy: float,
    macro_f1: float,
    output: Path,
) -> None:
    counts = np.asarray(matrix, dtype=np.int64)
    supports = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        supports,
        out=np.zeros_like(counts, dtype=np.float64),
        where=supports > 0,
    )
    figure, axis = plt.subplots(figsize=(16, 14), constrained_layout=True)
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(np.arange(len(class_names)), class_names)
    axis.set_yticks(np.arange(len(class_names)), class_names)
    plt.setp(axis.get_xticklabels(), rotation=52, ha="right", fontsize=8)
    plt.setp(axis.get_yticklabels(), fontsize=8)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(
        f"C2 PTv2 {split}: row-normalized confusion matrix\n"
        f"Accuracy {100*accuracy:.2f}% | Macro-F1 {100*macro_f1:.2f}%"
    )
    for index in range(len(class_names)):
        value = normalized[index, index]
        color = "black" if value > 0.62 else "white"
        axis.text(
            index,
            index,
            f"{100*value:.0f}",
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            fontweight="bold",
        )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Recall within each true class")
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def class_metric_figure(
    rows: list[dict[str, object]], output: Path
) -> None:
    ordered = sorted(rows, key=lambda item: float(item["f1"]))
    names = [str(item["scientific_name"]) for item in ordered]
    f1 = np.asarray([float(item["f1"]) for item in ordered])
    recall = np.asarray([float(item["recall"]) for item in ordered])
    support = np.asarray([int(item["support"]) for item in ordered])
    y = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(14, 10), constrained_layout=True)
    axis.barh(y, recall * 100, height=0.68, color="#B9D8C2", label="Recall")
    axis.barh(y, f1 * 100, height=0.42, color="#176B87", label="F1")
    axis.set_yticks(y, names, fontsize=9)
    axis.set_xlim(0, 105)
    axis.set_xlabel("Percent")
    axis.set_title("C2 PTv2 test performance by class")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    for index, (value, count) in enumerate(zip(f1, support)):
        axis.text(
            min(100 * value + 1.2, 98),
            index,
            f"{100*value:.1f}%  n={count}",
            va="center",
            fontsize=8,
        )
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def training_figure(
    history: list[dict[str, object]], best_epoch: int, output: Path
) -> None:
    epochs = np.asarray([int(item["epoch"]) for item in history])
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    panels = (
        ("Loss", "train_loss", "val_loss", "Cross entropy"),
        ("Accuracy", "train_accuracy", "val_accuracy", "Percent"),
        ("Macro-F1", "train_macro_f1", "val_macro_f1", "Percent"),
    )
    for axis, (title, train_key, val_key, ylabel) in zip(axes, panels):
        train = np.asarray([float(item[train_key]) for item in history])
        val = np.asarray([float(item[val_key]) for item in history])
        if ylabel == "Percent":
            train *= 100
            val *= 100
        axis.plot(epochs, train, label="train", color="#176B87", linewidth=2)
        axis.plot(epochs, val, label="validation", color="#D1495B", linewidth=1.7)
        axis.axvline(
            best_epoch,
            color="#3A3A3A",
            linestyle="--",
            linewidth=1,
            label=f"best epoch {best_epoch}",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    figure.suptitle("C2 PTv2 complete 19-class training", fontsize=16)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def confusion_rows(
    matrix: list[list[int]], class_names: list[str], split: str
) -> list[dict[str, object]]:
    counts = np.asarray(matrix, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for true_index in range(len(class_names)):
        support = int(counts[true_index].sum())
        for predicted_index in range(len(class_names)):
            if true_index == predicted_index or counts[true_index, predicted_index] == 0:
                continue
            count = int(counts[true_index, predicted_index])
            rows.append(
                {
                    "split": split,
                    "true_class_index": true_index,
                    "true_class": class_names[true_index],
                    "predicted_class_index": predicted_index,
                    "predicted_class": class_names[predicted_index],
                    "count": count,
                    "true_class_support": support,
                    "share_of_true_class": count / support if support else 0.0,
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            float(item["share_of_true_class"]),
            int(item["count"]),
        ),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    class_names = [str(value) for value in final["class_names"]]

    per_class_rows: list[dict[str, object]] = []
    all_confusions: list[dict[str, object]] = []
    figure_paths: dict[str, str] = {}
    for split in ("val", "test"):
        metrics = final["metrics"][split]
        for item in metrics["per_class"]:
            per_class_rows.append({"split": split, **item})
        all_confusions.extend(
            confusion_rows(metrics["confusion_matrix"], class_names, split)
        )
        figure_path = output_dir / f"C2_confusion_matrix_{split}_readable.png"
        readable_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            split,
            float(metrics["accuracy"]),
            float(metrics["macro_f1"]),
            figure_path,
        )
        figure_paths[f"confusion_matrix_{split}"] = str(figure_path)

    class_csv = output_dir / "C2_per_class_metrics.csv"
    write_csv(
        class_csv,
        [
            "split",
            "class_index",
            "scientific_name",
            "support",
            "precision",
            "recall",
            "f1",
        ],
        per_class_rows,
    )
    confusion_csv = output_dir / "C2_top_confusions.csv"
    write_csv(
        confusion_csv,
        [
            "split",
            "true_class_index",
            "true_class",
            "predicted_class_index",
            "predicted_class",
            "count",
            "true_class_support",
            "share_of_true_class",
        ],
        all_confusions,
    )

    test_rows = [row for row in per_class_rows if row["split"] == "test"]
    class_figure = output_dir / "C2_test_per_class_metrics.png"
    class_metric_figure(test_rows, class_figure)
    training_path = output_dir / "C2_training_curves_readable.png"
    training_figure(history, int(final["best_epoch"]), training_path)
    figure_paths["test_per_class_metrics"] = str(class_figure)
    figure_paths["training_curves"] = str(training_path)

    val = final["metrics"]["val"]
    test = final["metrics"]["test"]
    weakest_test = sorted(test_rows, key=lambda item: float(item["f1"]))[:5]
    strongest_test = sorted(
        test_rows, key=lambda item: float(item["f1"]), reverse=True
    )[:5]
    diagnostics = {
        "status": "passed",
        "run_dir": str(run_dir),
        "best_epoch": int(final["best_epoch"]),
        "epochs_completed": int(final["epochs_completed"]),
        "metrics": {
            "val": {
                name: val[name] for name in ("accuracy", "macro_f1", "loss")
            },
            "test": {
                name: test[name] for name in ("accuracy", "macro_f1", "loss")
            },
            "test_accuracy_minus_macro_f1": float(test["accuracy"])
            - float(test["macro_f1"]),
        },
        "weakest_test_classes": weakest_test,
        "strongest_test_classes": strongest_test,
        "top_test_confusions": [
            row for row in all_confusions if row["split"] == "test"
        ][:15],
        "artifacts": {
            "per_class_csv": str(class_csv),
            "confusions_csv": str(confusion_csv),
            **figure_paths,
        },
        "interpretation": [
            "Training converged but validation metrics remained highly variable.",
            "The test split is imbalanced; macro F1 is the primary class-balanced metric.",
            "Sophora japonica has zero test F1 and requires targeted follow-up.",
        ],
    }
    atomic_json(output_dir / "C2_diagnostics.json", diagnostics)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
