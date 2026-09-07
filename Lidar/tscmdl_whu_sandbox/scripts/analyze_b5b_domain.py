"""Aggregate B5/B5b seeds and diagnose fixed-road domain differences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import classification_metrics  # noqa: E402


GRID_SIZES = (0.06, 0.12, 0.24, 0.48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--run", action="append", type=Path, required=True, dest="runs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--metrics-plot", type=Path, required=True)
    parser.add_argument("--roads-plot", type=Path, required=True)
    parser.add_argument("--structure-plot", type=Path, required=True)
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


def load_predictions(path: Path) -> dict[str, dict[str, object]]:
    records = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["sample_key"]
            if key in records:
                raise ValueError(f"Duplicate prediction sample key: {key}")
            probabilities = [
                float(row[f"probability_{index}"]) for index in range(3)
            ]
            records[key] = {
                "split": row["split"],
                "true_class": int(row["true_class"]),
                "predicted_class": int(row["predicted_class"]),
                "probabilities": probabilities,
                "correct": bool(int(row["correct"])),
            }
    if len(records) != 180:
        raise ValueError(f"Expected 180 predictions, found {len(records)}")
    return records


def pooled_point_counts(points: np.ndarray) -> list[int]:
    coord = points.astype(np.float64, copy=False)
    counts = [int(coord.shape[0])]
    for grid_size in GRID_SIZES:
        origin = coord.min(axis=0, keepdims=True)
        voxel = np.floor((coord - origin) / grid_size).astype(np.int64)
        _, inverse, cluster_counts = np.unique(
            voxel, axis=0, return_inverse=True, return_counts=True
        )
        summed = np.zeros((len(cluster_counts), 3), dtype=np.float64)
        np.add.at(summed, inverse, coord)
        coord = summed / cluster_counts[:, None]
        counts.append(int(coord.shape[0]))
    return counts


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": math.nan, "std": math.nan}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def accuracy_for(
    keys: list[str], predictions: dict[str, dict[str, object]]
) -> float:
    if not keys:
        return math.nan
    return sum(bool(predictions[key]["correct"]) for key in keys) / len(keys)


def group_summary(
    keys: list[str],
    manifest: dict[str, dict[str, object]],
    predictions_by_seed: dict[int, dict[str, dict[str, object]]],
    ensemble: dict[str, dict[str, object]],
) -> dict[str, object]:
    class_histogram = Counter(int(manifest[key]["class_index"]) for key in keys)
    source_counts = [int(manifest[key]["source_point_count"]) for key in keys]
    stage_counts = [
        [int(value) for value in manifest[key]["b5b_stage_point_counts"]]
        for key in keys
    ]
    return {
        "sample_count": len(keys),
        "class_histogram": {
            str(key): value for key, value in sorted(class_histogram.items())
        },
        "source_point_count": {
            **describe([float(value) for value in source_counts]),
            "median": float(statistics.median(source_counts)),
        },
        "stage_point_count_mean": (
            np.asarray(stage_counts, dtype=np.float64).mean(axis=0).tolist()
            if stage_counts
            else []
        ),
        "accuracy_by_seed": {
            str(seed): accuracy_for(keys, predictions)
            for seed, predictions in predictions_by_seed.items()
        },
        "ensemble_accuracy": accuracy_for(keys, ensemble),
    }


def point_bin(source_count: int) -> str:
    if source_count < 8192:
        return "<8192"
    if source_count < 20000:
        return "8192-19999"
    if source_count < 50000:
        return "20000-49999"
    return ">=50000"


def stage4_bin(point_count: int) -> str:
    if point_count <= 8:
        return "<=8"
    if point_count <= 16:
        return "9-16"
    if point_count <= 24:
        return "17-24"
    return ">=25"


def plot_seed_metrics(
    path: Path,
    per_seed: dict[int, dict[str, object]],
    aggregate: dict[str, object],
    ensemble_metrics: dict[str, object],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = list(per_seed)
    x = np.arange(len(seeds))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for axis, split in zip(axes, ("val", "test")):
        accuracy = [
            100 * per_seed[seed]["metrics"][split]["accuracy"] for seed in seeds
        ]
        macro_f1 = [
            100 * per_seed[seed]["metrics"][split]["macro_f1"] for seed in seeds
        ]
        axis.plot(x, accuracy, "o-", label="Accuracy", color="#1565C0")
        axis.plot(x, macro_f1, "s-", label="Macro F1", color="#D84315")
        axis.axhline(
            100 * aggregate[split]["macro_f1"]["mean"],
            color="#D84315",
            linestyle="--",
            alpha=0.55,
            label="Mean Macro F1",
        )
        axis.scatter(
            [len(seeds) - 0.5],
            [100 * ensemble_metrics[split]["macro_f1"]],
            marker="*",
            s=150,
            color="#2E7D32",
            label="Probability ensemble",
            zorder=5,
        )
        axis.set_xticks(
            list(x) + [len(seeds) - 0.5],
            [str(seed)[-2:] for seed in seeds] + ["ens"],
        )
        axis.set_title(split.upper())
        axis.set_xlabel("Seed suffix")
        axis.set_ylabel("Percent")
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="lower right")
    figure.suptitle("PTv2 repeatability on the fixed B1 split")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_roads(
    path: Path,
    road_summaries: dict[str, dict[str, object]],
    train_roads: set[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for split in ("val", "test"):
        for road, summary in road_summaries[split].items():
            rows.append(
                (
                    split,
                    road,
                    int(summary["sample_count"]),
                    100
                    * statistics.mean(summary["accuracy_by_seed"].values()),
                    road in train_roads,
                )
            )
    labels = [f"{split}:{road}" for split, road, *_ in rows]
    counts = [row[2] for row in rows]
    accuracy = [row[3] for row in rows]
    colors = ["#00897B" if row[4] else "#EF6C00" for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    axes[0].bar(x, counts, color=colors)
    axes[0].set_ylabel("Samples")
    axes[0].set_title("Road composition (green=seen in train, orange=unseen)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, accuracy, color=colors)
    axes[1].set_ylabel("Mean seed accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, labels, rotation=55, ha="right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_structure(
    path: Path,
    eval_records: list[dict[str, object]],
    point_bins: dict[str, dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for split, color in (("val", "#D84315"), ("test", "#1565C0")):
        rows = [row for row in eval_records if row["split"] == split]
        axes[0].scatter(
            [row["source_point_count"] for row in rows],
            [row["stage1_point_count"] for row in rows],
            s=24,
            alpha=0.7,
            label=split,
            color=color,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Source point count (log scale)")
    axes[0].set_ylabel("Stage-1 grid point count")
    axes[0].set_title("Sampling density and grid retention")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    order = ["<8192", "8192-19999", "20000-49999", ">=50000"]
    x = np.arange(len(order))
    width = 0.36
    for index, split in enumerate(("val", "test")):
        values = [
            100 * point_bins[split][label]["mean_seed_accuracy"]
            if label in point_bins[split]
            else math.nan
            for label in order
        ]
        axes[1].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=split,
            color=("#D84315", "#1565C0")[index],
        )
    axes[1].set_xticks(x, order, rotation=20, ha="right")
    axes[1].set_ylabel("Mean seed accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Accuracy by source point-count bin")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    run_dirs = [path.resolve() for path in args.runs]
    if len(run_dirs) != 3:
        raise ValueError("B5b analysis requires exactly three seed runs")

    manifest_payload = read_json(dataset_root / "manifest.json")
    records = {
        str(record["sample_key"]): dict(record)
        for record in manifest_payload["records"]
    }
    class_names = [
        item["scientific_name"]
        for item in sorted(
            read_json(dataset_root / "classes.json"),
            key=lambda item: int(item["class_index"]),
        )
    ]
    eval_keys = sorted(
        key for key, record in records.items() if record["split"] in {"val", "test"}
    )
    for index, key in enumerate(eval_keys, start=1):
        record = records[key]
        point_path = dataset_root / record["point_path"]
        with np.load(point_path, allow_pickle=False) as archive:
            points = archive["points_xyz"]
        record["b5b_stage_point_counts"] = pooled_point_counts(points)
        if index % 30 == 0 or index == len(eval_keys):
            print(
                f"\rgeometry [{index:3d}/{len(eval_keys):3d}]",
                end="",
                flush=True,
            )
    print(flush=True)

    per_seed: dict[int, dict[str, object]] = {}
    predictions_by_seed: dict[int, dict[str, dict[str, object]]] = {}
    locked_signature = None
    for run_dir in run_dirs:
        config = read_json(run_dir / "run_config.json")
        final = read_json(run_dir / "final_metrics.json")
        seed = int(config["args"]["seed"])
        if seed in per_seed:
            raise ValueError(f"Duplicate seed run: {seed}")
        signature = {
            key: value
            for key, value in config["args"].items()
            if key not in {"seed", "output_dir", "resume"}
        }
        signature["model_config"] = config["model_config"]
        if locked_signature is None:
            locked_signature = signature
        elif signature != locked_signature:
            raise ValueError(f"Run configuration drift for seed {seed}")
        predictions = load_predictions(run_dir / "predictions.csv")
        if set(predictions) != set(eval_keys):
            raise ValueError(f"Prediction sample set mismatch for seed {seed}")
        predictions_by_seed[seed] = predictions
        per_seed[seed] = {
            "run_dir": str(run_dir),
            "best_epoch": int(final["best_epoch"]),
            "epochs_completed": int(final["epochs_completed"]),
            "training_session_elapsed_seconds": float(
                final["training_session_elapsed_seconds"]
            ),
            "metrics": {
                split: {
                    name: float(final["metrics"][split][name])
                    for name in (
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                    )
                }
                for split in ("val", "test")
            },
        }
    per_seed = dict(sorted(per_seed.items()))
    predictions_by_seed = dict(sorted(predictions_by_seed.items()))
    seeds = list(per_seed)

    ensemble: dict[str, dict[str, object]] = {}
    for key in eval_keys:
        probabilities = np.asarray(
            [predictions_by_seed[seed][key]["probabilities"] for seed in seeds],
            dtype=np.float64,
        ).mean(axis=0)
        true_class = int(records[key]["class_index"])
        predicted_class = int(np.argmax(probabilities))
        ensemble[key] = {
            "split": records[key]["split"],
            "true_class": true_class,
            "predicted_class": predicted_class,
            "probabilities": probabilities.tolist(),
            "correct": predicted_class == true_class,
        }

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    ensemble_metrics: dict[str, dict[str, object]] = {}
    for split in ("val", "test"):
        aggregate[split] = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = [
                float(per_seed[seed]["metrics"][split][metric]) for seed in seeds
            ]
            aggregate[split][metric] = describe(values)
        keys = [key for key in eval_keys if records[key]["split"] == split]
        ensemble_metrics[split] = classification_metrics(
            [int(records[key]["class_index"]) for key in keys],
            [int(ensemble[key]["predicted_class"]) for key in keys],
            len(class_names),
        )

    pairwise_agreement = {}
    for left, right in combinations(seeds, 2):
        pairwise_agreement[f"{left}-{right}"] = {
            split: float(
                np.mean(
                    [
                        predictions_by_seed[left][key]["predicted_class"]
                        == predictions_by_seed[right][key]["predicted_class"]
                        for key in eval_keys
                        if records[key]["split"] == split
                    ]
                )
            )
            for split in ("val", "test")
        }

    train_roads = {
        str(record["road_id"])
        for record in records.values()
        if record["split"] == "train"
    }
    roads_by_split = {
        split: {
            str(records[key]["road_id"])
            for key in eval_keys
            if records[key]["split"] == split
        }
        for split in ("val", "test")
    }
    road_summaries: dict[str, dict[str, object]] = {}
    for split in ("val", "test"):
        road_summaries[split] = {}
        for road in sorted(roads_by_split[split]):
            keys = [
                key
                for key in eval_keys
                if records[key]["split"] == split
                and str(records[key]["road_id"]) == road
            ]
            summary = group_summary(
                keys, records, predictions_by_seed, ensemble
            )
            summary["seen_in_train"] = road in train_roads
            road_summaries[split][road] = summary

    class_summaries = {
        split: {
            str(class_index): {
                **group_summary(
                    [
                        key
                        for key in eval_keys
                        if records[key]["split"] == split
                        and int(records[key]["class_index"]) == class_index
                    ],
                    records,
                    predictions_by_seed,
                    ensemble,
                ),
                "scientific_name": class_names[class_index],
            }
            for class_index in range(len(class_names))
        }
        for split in ("val", "test")
    }

    point_bins: dict[str, dict[str, object]] = {"val": {}, "test": {}}
    stage4_bins: dict[str, dict[str, object]] = {"val": {}, "test": {}}
    for split in ("val", "test"):
        split_keys = [key for key in eval_keys if records[key]["split"] == split]
        for label in ("<8192", "8192-19999", "20000-49999", ">=50000"):
            keys = [
                key
                for key in split_keys
                if point_bin(int(records[key]["source_point_count"])) == label
            ]
            if keys:
                summary = group_summary(
                    keys, records, predictions_by_seed, ensemble
                )
                summary["mean_seed_accuracy"] = statistics.mean(
                    summary["accuracy_by_seed"].values()
                )
                point_bins[split][label] = summary
        for label in ("<=8", "9-16", "17-24", ">=25"):
            keys = [
                key
                for key in split_keys
                if stage4_bin(int(records[key]["b5b_stage_point_counts"][4])) == label
            ]
            if keys:
                summary = group_summary(
                    keys, records, predictions_by_seed, ensemble
                )
                summary["mean_seed_accuracy"] = statistics.mean(
                    summary["accuracy_by_seed"].values()
                )
                stage4_bins[split][label] = summary

    eval_rows = []
    for key in eval_keys:
        record = records[key]
        row = {
            "sample_key": key,
            "split": record["split"],
            "road_id": record["road_id"],
            "road_seen_in_train": int(str(record["road_id"]) in train_roads),
            "class_index": int(record["class_index"]),
            "scientific_name": record["scientific_name"],
            "source_point_count": int(record["source_point_count"]),
            "sampled_unique_point_count": int(record["sampled_unique_point_count"]),
        }
        for stage, value in enumerate(record["b5b_stage_point_counts"]):
            row[f"stage{stage}_point_count"] = int(value)
        for seed in seeds:
            prediction = predictions_by_seed[seed][key]
            row[f"seed{seed}_predicted_class"] = prediction["predicted_class"]
            row[f"seed{seed}_correct"] = int(prediction["correct"])
        row["ensemble_predicted_class"] = ensemble[key]["predicted_class"]
        row["ensemble_correct"] = int(ensemble[key]["correct"])
        eval_rows.append(row)

    args.sample_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = Path(f"{args.sample_csv}.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(eval_rows[0]))
        writer.writeheader()
        writer.writerows(eval_rows)
    temporary_csv.replace(args.sample_csv)

    result = {
        "status": "passed",
        "stage": "B5b",
        "seeds": seeds,
        "class_names": class_names,
        "locked_configuration": locked_signature,
        "per_seed": {str(seed): value for seed, value in per_seed.items()},
        "aggregate": aggregate,
        "ensemble_metrics": ensemble_metrics,
        "pairwise_prediction_agreement": pairwise_agreement,
        "road_split_structure": {
            "train_roads": sorted(train_roads),
            "val_roads": sorted(roads_by_split["val"]),
            "test_roads": sorted(roads_by_split["test"]),
            "val_seen_in_train": sorted(roads_by_split["val"] & train_roads),
            "val_unseen_in_train": sorted(roads_by_split["val"] - train_roads),
            "test_seen_in_train": sorted(roads_by_split["test"] & train_roads),
            "test_unseen_in_train": sorted(roads_by_split["test"] - train_roads),
            "val_samples_on_seen_roads": sum(
                str(records[key]["road_id"]) in train_roads
                for key in eval_keys
                if records[key]["split"] == "val"
            ),
            "test_samples_on_seen_roads": sum(
                str(records[key]["road_id"]) in train_roads
                for key in eval_keys
                if records[key]["split"] == "test"
            ),
        },
        "road_summaries": road_summaries,
        "class_summaries": class_summaries,
        "source_point_count_bins": point_bins,
        "stage4_point_count_bins": stage4_bins,
        "artifacts": {
            "sample_csv": str(args.sample_csv.resolve()),
            "metrics_plot": str(args.metrics_plot.resolve()),
            "roads_plot": str(args.roads_plot.resolve()),
            "structure_plot": str(args.structure_plot.resolve()),
        },
    }
    atomic_json(args.output, result)
    plot_seed_metrics(
        args.metrics_plot, per_seed, aggregate, ensemble_metrics
    )
    plot_roads(args.roads_plot, road_summaries, train_roads)
    plot_structure(args.structure_plot, eval_rows, point_bins)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
