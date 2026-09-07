"""Summarize completed C3 seeds into paper-ready aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "c3_config.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    rows: list[dict[str, Any]] = []
    final_by_seed: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        path = output / "runs" / f"seed_{seed}" / "final_metrics.json"
        if not path.is_file():
            continue
        final = json.loads(path.read_text(encoding="utf-8-sig"))
        if final.get("status") != "complete":
            continue
        final_by_seed[seed] = final
        val = final["metrics"]["val"]
        test = final["metrics"]["test"]
        rows.append(
            {
                "seed": seed,
                "best_epoch": int(final["best_epoch"]),
                "epochs_completed": int(final["epochs_completed"]),
                "elapsed_seconds": float(final["training_elapsed_seconds"]),
                "val_accuracy": float(val["accuracy"]),
                "val_balanced_accuracy": float(val["balanced_accuracy"]),
                "val_macro_f1": float(val["macro_f1"]),
                "test_accuracy": float(test["accuracy"]),
                "test_balanced_accuracy": float(test["balanced_accuracy"]),
                "test_macro_f1": float(test["macro_f1"]),
            }
        )

    metric_fields = [
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    ]
    aggregate = {
        metric: mean_std([float(row[metric]) for row in rows])
        for metric in metric_fields
    }
    per_class_rows: list[dict[str, Any]] = []
    if final_by_seed:
        first = next(iter(final_by_seed.values()))
        class_names = list(first["class_names"])
        for index, name in enumerate(class_names):
            f1_values = []
            recall_values = []
            for seed in seeds:
                if seed not in final_by_seed:
                    continue
                metrics = final_by_seed[seed]["metrics"]["test"]["per_class"][index]
                f1_values.append(float(metrics["f1"]))
                recall_values.append(float(metrics["recall"]))
            f1 = mean_std(f1_values)
            recall = mean_std(recall_values)
            per_class_rows.append(
                {
                    "class_index": index,
                    "scientific_name": name,
                    "completed_seeds": len(f1_values),
                    "test_f1_mean": f1["mean"],
                    "test_f1_std": f1["std"],
                    "test_recall_mean": recall["mean"],
                    "test_recall_std": recall["std"],
                }
            )

    status = "complete" if len(rows) == len(seeds) else "partial"
    summary = {
        "schema_version": 1,
        "stage": "C3",
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fixed_config": config,
        "completed_seeds": [int(row["seed"]) for row in rows],
        "pending_seeds": [seed for seed in seeds if seed not in final_by_seed],
        "seed_metrics": rows,
        "aggregate": aggregate,
        "per_class_stability": per_class_rows,
    }
    atomic_json(output / "c3_summary.json", summary)
    write_csv(
        output / "c3_seed_metrics.csv",
        rows,
        [
            "seed",
            "best_epoch",
            "epochs_completed",
            "elapsed_seconds",
            *metric_fields,
        ],
    )
    write_csv(
        output / "c3_per_class_stability.csv",
        per_class_rows,
        [
            "class_index",
            "scientific_name",
            "completed_seeds",
            "test_f1_mean",
            "test_f1_std",
            "test_recall_mean",
            "test_recall_std",
        ],
    )

    if rows:
        labels = [str(row["seed"]) for row in rows]
        x = list(range(len(rows)))
        width = 0.2
        fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        series = [
            ("Val accuracy", "val_accuracy"),
            ("Val macro-F1", "val_macro_f1"),
            ("Test accuracy", "test_accuracy"),
            ("Test macro-F1", "test_macro_f1"),
        ]
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        colors = ["#2878B5", "#9AC9DB", "#C82423", "#F8AC8C"]
        for (label, field), offset, color in zip(series, offsets, colors):
            axis.bar(
                [value + offset for value in x],
                [100.0 * float(row[field]) for row in rows],
                width,
                label=label,
                color=color,
            )
        axis.set_xticks(x, labels)
        axis.set_xlabel("Random seed")
        axis.set_ylabel("Score (%)")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=2)
        axis.set_title("C3 PTv2 fixed-configuration repeat metrics")
        fig.savefig(output / "c3_seed_metrics.png", dpi=180)
        plt.close(fig)

    def metric_text(name: str) -> str:
        values = aggregate[name]
        return f"{100*values['mean']:.2f}% +/- {100*values['std']:.2f}%"

    lines = [
        "# C3 三随机种子结果摘要",
        "",
        f"状态：{status}",
        f"完成种子：{len(rows)}/{len(seeds)}",
        "",
        "## 固定配置",
        "",
        f"- 随机种子：{', '.join(str(seed) for seed in seeds)}",
        f"- batch/eval batch：{config['batch_size']}/{config['eval_batch_size']}",
        f"- 最大轮次/patience：{config['epochs']}/{config['patience']}",
        "- 每个种子均从零开始，使用同一数据划分与超参数。",
        "",
        "## 汇总指标",
        "",
        f"- Validation Accuracy：{metric_text('val_accuracy')}",
        f"- Validation Macro-F1：{metric_text('val_macro_f1')}",
        f"- Test Accuracy：{metric_text('test_accuracy')}",
        f"- Test Macro-F1：{metric_text('test_macro_f1')}",
        "",
        "均值与标准差仅基于当前已完成种子；三个种子全部完成后才是正式 C3 结果。",
    ]
    atomic_text(output / "C3_三随机种子结果摘要.md", "\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
