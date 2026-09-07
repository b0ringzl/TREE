"""Compare aligned B2, B3, B4, and B5 classification predictions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.status import (  # noqa: E402
    compare_prediction_models,
    load_prediction_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2-run", type=Path, required=True)
    parser.add_argument("--b3-run", type=Path, required=True)
    parser.add_argument("--b4-run", type=Path, required=True)
    parser.add_argument("--b5-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def metric_subset(final: dict[str, object], split: str) -> dict[str, float]:
    metrics = final["metrics"][split]
    return {
        name: float(metrics[name])
        for name in ("accuracy", "balanced_accuracy", "macro_f1")
    }


def plot_metrics(path: Path, metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(metrics)
    labels = ["PointMLP", "ResNet50", "TSCMDL", "PTv2"]
    colors = ["#1565C0", "#00897B", "#EF6C00", "#7B1FA2"]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, split in zip(axes, ("val", "test")):
        accuracy = [100 * metrics[name][split]["accuracy"] for name in names]
        macro_f1 = [100 * metrics[name][split]["macro_f1"] for name in names]
        width = 0.36
        axis.bar(x - width / 2, accuracy, width, label="Accuracy", color=colors)
        axis.bar(
            x + width / 2,
            macro_f1,
            width,
            label="Macro F1",
            color=colors,
            alpha=0.55,
            hatch="//",
        )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_title(split.upper())
        axis.set_ylim(0, 100)
        axis.set_ylabel("Percent")
        axis.grid(axis="y", alpha=0.25)
        for index, (left, right) in enumerate(zip(accuracy, macro_f1)):
            axis.text(index - width / 2, left + 1.0, f"{left:.1f}", ha="center")
            axis.text(index + width / 2, right + 1.0, f"{right:.1f}", ha="center")
    axes[0].legend(loc="lower right")
    figure.suptitle("Three-species baseline comparison on the fixed B1 split")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    runs = {
        "B2": args.b2_run.resolve(),
        "B3": args.b3_run.resolve(),
        "B4": args.b4_run.resolve(),
        "B5": args.b5_run.resolve(),
    }
    predictions = {
        name: load_prediction_records(run / "predictions.csv")
        for name, run in runs.items()
    }
    finals = {
        name: read_json(run / "final_metrics.json")
        for name, run in runs.items()
    }
    class_names = finals["B2"]["class_names"]
    for name, final in finals.items():
        if final["class_names"] != class_names:
            raise ValueError(f"Class-name mismatch for {name}")
    metrics = {
        name: {
            split: metric_subset(final, split)
            for split in ("val", "test")
        }
        for name, final in finals.items()
    }
    overlap = compare_prediction_models(predictions, len(class_names))
    result = {
        "status": "passed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "class_names": class_names,
        "runs": {name: str(path) for name, path in runs.items()},
        "metrics": metrics,
        "correctness_overlap": overlap,
    }
    atomic_json(args.output, result)
    plot_metrics(args.plot, metrics)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
