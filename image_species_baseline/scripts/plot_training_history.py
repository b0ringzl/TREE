#!/usr/bin/env python3
"""Plot loss and HK validation metrics from a completed training history."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "training_history.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Training history is empty")
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    crop_f1 = [float(row["val_crop_macro_f1"]) for row in rows]
    group_f1 = [float(row["val_group_macro_f1"]) for row in rows]
    best_index = max(range(len(rows)), key=lambda index: group_f1[index])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, crop_f1, label="HK crop macro-F1", linewidth=2)
    axes[1].plot(epochs, group_f1, label="HK tree-group macro-F1", linewidth=2)
    axes[1].scatter(
        [epochs[best_index]],
        [group_f1[best_index]],
        color="#d62728",
        zorder=3,
        label=f"best epoch {epochs[best_index]}",
    )
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(run_dir / "training_curves.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
