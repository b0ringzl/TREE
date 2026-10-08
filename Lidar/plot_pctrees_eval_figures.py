import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

TREE_ROOT = Path(__file__).resolve().parent.parent
PCTREES_ROOT = TREE_ROOT / "Lidar" / "pctrees_github"
LIDAR_DATA_ROOT = TREE_ROOT / "lidar data"


def load_class_names(pctrees_root, label_path, data_dir, cache_dir, min_points, num_points, train_split):
    import sys

    pctrees_root = Path(pctrees_root)
    sys.path.insert(0, str(pctrees_root))
    from pct_main import load_tree_lidar_data

    args = SimpleNamespace(
        label_path=str(label_path),
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        min_points=min_points,
        num_points=num_points,
        train_split=train_split,
        top_species=None,
    )
    _, val_dataset = load_tree_lidar_data(args)
    dataset = getattr(val_dataset, "dataset", val_dataset)
    return list(dataset.classes)


def save_confusion_matrix(matrix, class_names, out_path, title, normalize=False):
    fig_width = max(8, len(class_names) * 1.25)
    fig_height = max(6, len(class_names) * 1.05)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    threshold = np.nanmax(matrix) / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if normalize:
                text = f"{matrix[i, j]:.2f}"
            else:
                text = str(int(matrix[i, j]))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_per_class_metrics(report_df, out_path):
    metric_cols = ["precision", "recall", "f1-score"]
    plot_df = report_df.loc[
        [idx for idx in report_df.index if idx not in {"accuracy", "macro avg", "weighted avg"}],
        metric_cols,
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.35), 5.5), dpi=180)
    x = np.arange(len(plot_df.index))
    width = 0.25
    for offset, metric in zip([-width, 0, width], metric_cols):
        ax.bar(x + offset, plot_df[metric].values, width, label=metric)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class validation metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=35, ha="right", fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overall_summary(metrics, out_path):
    labels = ["Accuracy", "Balanced accuracy", "Macro F1", "Weighted F1"]
    values = [
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
    bars = ax.bar(labels, values, color=["#2f6f9f", "#3f8f72", "#b36b3c", "#7a5aa6"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall validation accuracy summary")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_dashboard(cm_norm, class_names, metrics, out_path, title):
    class_count = len(class_names)
    dense_matrix = class_count > 15
    figure_height = max(7.5, class_count * 0.38) if dense_matrix else 5.2
    figure_width = max(17.5, class_count * 0.55) if dense_matrix else 13.5
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        dpi=180,
        gridspec_kw={"width_ratios": [1.65, 0.9] if dense_matrix else [1.25, 1]},
        constrained_layout=True,
    )

    ax = axes[0]
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    tick_fontsize = 7 if dense_matrix else 8
    ax.set_xticklabels(
        class_names,
        rotation=55 if dense_matrix else 35,
        ha="right",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(class_names, fontsize=tick_fontsize)
    if not dense_matrix:
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8,
                )

    ax = axes[1]
    labels = ["Accuracy", "Balanced\naccuracy", "Macro F1", "Weighted F1"]
    values = [
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    ]
    bars = ax.bar(labels, values, color=["#2f6f9f", "#3f8f72", "#b36b3c", "#7a5aa6"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation summary")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        0.5,
        -0.22,
        f"Validation samples: {metrics['sample_count']} | Classes: {metrics['class_count']}",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        default=str(LIDAR_DATA_ROOT / "experiments" / "five_species_500_train_25_val_20260701"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(PCTREES_ROOT / "checkpoints" / "five_species_500_train_25_val_20260701_5ep_full500"),
    )
    parser.add_argument("--pctrees-root", default=str(PCTREES_ROOT))
    parser.add_argument("--data-dir", default=str(LIDAR_DATA_ROOT / "train"))
    parser.add_argument("--label-path", help="Validation label CSV. Defaults to the five-species experiment labels.")
    parser.add_argument("--cache-dir", help="Point tensor cache directory. Defaults to the five-species experiment cache.")
    parser.add_argument("--output-dir", help="Directory for figures and CSV/JSON reports. Defaults to <experiment-dir>/figures.")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--min-points", type=int, default=1024)
    parser.add_argument("--pred-pickle", default=str(PCTREES_ROOT / "pct_preds.pickle"))
    parser.add_argument("--true-pickle", default=str(PCTREES_ROOT / "labels.pickle"))
    parser.add_argument("--dashboard-title", default="PCTree validation overview")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    out_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = Path(args.label_path) if args.label_path else experiment_dir / "labels_5species_525_split.csv"
    cache_dir = Path(args.cache_dir) if args.cache_dir else experiment_dir / "point_cache_4096"

    with open(args.true_pickle, "rb") as f:
        y_true = np.asarray(pickle.load(f), dtype=int).reshape(-1)
    with open(args.pred_pickle, "rb") as f:
        y_pred = np.asarray(pickle.load(f), dtype=int).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    class_names = load_class_names(
        args.pctrees_root,
        label_path,
        args.data_dir,
        cache_dir,
        args.min_points,
        args.num_points,
        args.train_split,
    )
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    metrics = {
        "sample_count": int(len(y_true)),
        "class_count": int(len(class_names)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(report_df.loc["macro avg", "f1-score"]),
        "weighted_f1": float(report_df.loc["weighted avg", "f1-score"]),
        "class_names": class_names,
    }

    save_confusion_matrix(
        cm,
        class_names,
        out_dir / "confusion_matrix_counts.png",
        "Validation confusion matrix (counts)",
        normalize=False,
    )
    save_confusion_matrix(
        cm_norm,
        class_names,
        out_dir / "confusion_matrix_normalized.png",
        "Validation confusion matrix (row-normalized)",
        normalize=True,
    )
    save_per_class_metrics(report_df, out_dir / "per_class_metrics.png")
    save_overall_summary(metrics, out_dir / "overall_accuracy_summary.png")
    save_dashboard(cm_norm, class_names, metrics, out_dir / "validation_dashboard.png", args.dashboard_title)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / "confusion_matrix_counts.csv", encoding="utf-8-sig", index_label="true_class"
    )
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        out_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig", index_label="true_class"
    )
    report_df.to_csv(out_dir / "classification_report.csv", encoding="utf-8-sig", index_label="class")
    pd.DataFrame(
        {
            "true_id": y_true,
            "pred_id": y_pred,
            "true_class": [class_names[i] for i in y_true],
            "pred_class": [class_names[i] for i in y_pred],
            "correct": y_true == y_pred,
        }
    ).to_csv(out_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": str(out_dir), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
