"""Create a road-exclusive cross-validation protocol from a WHU inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    build_road_class_counts,
    choose_inner_validation_roads,
    plan_outer_road_folds,
    select_balanced_evaluation_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--eval-per-class", type=int, default=30)
    parser.add_argument("--min-train-per-class", type=int, default=30)
    parser.add_argument("--validation-road-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--restarts", type=int, default=2000)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def histogram(
    instances: list[dict[str, object]], labels: list[int]
) -> dict[str, int]:
    counts = Counter(int(item["benchmark_label_id"]) for item in instances)
    return {str(label): counts[label] for label in labels}


def plot_protocol(
    path: Path,
    labels: list[int],
    class_names: dict[int, str],
    fold_summaries: list[dict[str, object]],
    run_summaries: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#2A6F97", "#E07A5F", "#3D8D7A"]
    x = np.arange(len(labels))
    width = 0.23
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    for fold, summary in enumerate(fold_summaries):
        values = [summary["class_histogram"][str(label)] for label in labels]
        axes[0].bar(
            x + (fold - 1) * width,
            values,
            width,
            label=f"Fold {fold + 1}",
            color=colors[fold % len(colors)],
        )
    axes[0].set_yscale("log")
    axes[0].set_title("All development samples by outer fold")
    axes[0].set_ylabel("Samples (log scale)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [class_names[label].split()[0] for label in labels],
        rotation=20,
        ha="right",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    selected = [
        run["balanced_evaluation_histogram"]["test"] for run in run_summaries
    ]
    for run_index, values_by_label in enumerate(selected):
        values = [values_by_label[str(label)] for label in labels]
        axes[1].bar(
            x + (run_index - 1) * width,
            values,
            width,
            label=f"Run {run_index + 1}",
            color=colors[run_index % len(colors)],
        )
    axes[1].set_title("Primary balanced road-unseen test set")
    axes[1].set_ylabel("Selected samples")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [class_names[label].split()[0] for label in labels],
        rotation=20,
        ha="right",
    )
    axes[1].set_ylim(0, max(max(row.values()) for row in selected) * 1.2)
    axes[1].grid(axis="y", alpha=0.2)

    split_names = ("train", "val", "test")
    run_x = np.arange(len(run_summaries))
    for split_index, split in enumerate(split_names):
        axes[2].bar(
            run_x + (split_index - 1) * width,
            [run["road_counts"][split] for run in run_summaries],
            width,
            label=split,
            color=colors[split_index],
        )
    axes[2].set_title("Whole-road assignments per CV run")
    axes[2].set_ylabel("Roads")
    axes[2].set_xticks(run_x)
    axes[2].set_xticklabels([f"Run {index + 1}" for index in run_x])
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)

    figure.suptitle("B5c road-grouped cross-validation protocol", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_road_fraction < 1.0:
        raise ValueError("validation-road-fraction must be between 0 and 1")

    inventory = read_json(args.inventory)
    classes = read_json(args.classes)
    if not isinstance(inventory, dict) or not isinstance(classes, list):
        raise ValueError("Invalid inventory or classes file")
    if inventory["summary"]["completed_trajectory_count"] != inventory["trajectory_count"]:
        raise ValueError("Inventory is incomplete")

    labels = [int(item["benchmark_label_id"]) for item in classes]
    class_names = {
        int(item["benchmark_label_id"]): str(item["scientific_name"])
        for item in classes
    }
    all_instances = [
        item
        for item in inventory["instances"]
        if int(item["benchmark_label_id"]) in set(labels)
    ]
    development = [item for item in all_instances if not item["is_official_test"]]
    official_test = [item for item in all_instances if item["is_official_test"]]
    road_counts = build_road_class_counts(development, labels)
    minimum = {label: args.eval_per_class for label in labels}
    train_minimum = {label: args.min_train_per_class for label in labels}
    outer = plan_outer_road_folds(
        road_counts,
        labels,
        n_splits=args.folds,
        minimum_per_class=minimum,
        seed=args.seed,
        restarts=args.restarts,
    )

    output_dir = args.output_dir.resolve()
    protocol_path = output_dir / "road_cv_protocol.json"
    road_counts_path = output_dir / "road_class_counts.csv"
    road_assignments_path = output_dir / "road_assignments.csv"
    sample_assignments_path = output_dir / "sample_assignments.csv"
    validation_path = output_dir / "validation.json"
    figure_path = output_dir / "road_cv_protocol.png"

    road_count_rows = []
    for road in sorted(road_counts):
        row: dict[str, object] = {
            "road_id": road,
            "sample_count": sum(road_counts[road].values()),
            "outer_fold": outer[road],
        }
        for label in labels:
            row[f"label_{label}_count"] = road_counts[road][label]
        road_count_rows.append(row)

    road_assignment_rows = []
    sample_assignment_rows = []
    fold_summaries = []
    run_summaries = []
    development_roads = set(road_counts)

    for fold in range(args.folds):
        fold_roads = {road for road, assigned in outer.items() if assigned == fold}
        fold_instances = [
            item for item in development if str(item["road_id"]) in fold_roads
        ]
        fold_summaries.append(
            {
                "fold": fold,
                "roads": sorted(fold_roads),
                "road_count": len(fold_roads),
                "sample_count": len(fold_instances),
                "class_histogram": histogram(fold_instances, labels),
            }
        )

    for run_index in range(args.folds):
        test_roads = {
            road for road, assigned in outer.items() if assigned == run_index
        }
        remaining_count = len(development_roads - test_roads)
        validation_road_count = max(
            1, round(remaining_count * args.validation_road_fraction)
        )
        val_roads = set(
            choose_inner_validation_roads(
                road_counts,
                labels,
                test_roads,
                validation_minimum=minimum,
                training_minimum=train_minimum,
                target_road_count=validation_road_count,
                seed=args.seed + run_index,
            )
        )
        train_roads = development_roads - test_roads - val_roads
        split_roads = {
            "train": train_roads,
            "val": val_roads,
            "test": test_roads,
        }
        if any(
            split_roads[left] & split_roads[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError(f"Road leakage detected in run {run_index}")

        split_instances = {
            split: [
                item
                for item in development
                if str(item["road_id"]) in roads
            ]
            for split, roads in split_roads.items()
        }
        evaluation_keys = {
            split: select_balanced_evaluation_keys(
                split_instances[split],
                labels,
                per_class=args.eval_per_class,
                seed=args.seed,
                namespace=f"run-{run_index}-{split}",
            )
            for split in ("val", "test")
        }

        for road in sorted(development_roads):
            split = next(
                name for name, roads in split_roads.items() if road in roads
            )
            row = {
                "run_index": run_index,
                "road_id": road,
                "outer_fold": outer[road],
                "split": split,
                "sample_count": sum(road_counts[road].values()),
            }
            for label in labels:
                row[f"label_{label}_count"] = road_counts[road][label]
            road_assignment_rows.append(row)

        for split in ("train", "val", "test"):
            for item in split_instances[split]:
                key = str(item["sample_key"])
                sample_assignment_rows.append(
                    {
                        "run_index": run_index,
                        "sample_key": key,
                        "split": split,
                        "road_id": str(item["road_id"]),
                        "trajectory_id": str(item["trajectory_id"]),
                        "tree_id": int(item["tree_id"]),
                        "benchmark_label_id": int(item["benchmark_label_id"]),
                        "scientific_name": class_names[
                            int(item["benchmark_label_id"])
                        ],
                        "balanced_evaluation_selected": int(
                            split in evaluation_keys
                            and key in evaluation_keys[split]
                        ),
                        "is_official_test": 0,
                    }
                )

        run_summaries.append(
            {
                "run_index": run_index,
                "outer_test_fold": run_index,
                "roads": {
                    split: sorted(roads) for split, roads in split_roads.items()
                },
                "road_counts": {
                    split: len(roads) for split, roads in split_roads.items()
                },
                "sample_counts": {
                    split: len(rows) for split, rows in split_instances.items()
                },
                "class_histograms": {
                    split: histogram(rows, labels)
                    for split, rows in split_instances.items()
                },
                "balanced_evaluation_histogram": {
                    split: {
                        str(label): sum(
                            int(item["benchmark_label_id"]) == label
                            and str(item["sample_key"]) in evaluation_keys[split]
                            for item in split_instances[split]
                        )
                        for label in labels
                    }
                    for split in ("val", "test")
                },
                "road_overlap_count": 0,
            }
        )

    atomic_csv(
        road_counts_path,
        road_count_rows,
        ["road_id", "sample_count", "outer_fold"]
        + [f"label_{label}_count" for label in labels],
    )
    atomic_csv(
        road_assignments_path,
        road_assignment_rows,
        ["run_index", "road_id", "outer_fold", "split", "sample_count"]
        + [f"label_{label}_count" for label in labels],
    )
    atomic_csv(
        sample_assignments_path,
        sample_assignment_rows,
        [
            "run_index",
            "sample_key",
            "split",
            "road_id",
            "trajectory_id",
            "tree_id",
            "benchmark_label_id",
            "scientific_name",
            "balanced_evaluation_selected",
            "is_official_test",
        ],
    )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    protocol = {
        "schema_version": 1,
        "stage": "B5c",
        "generated_at": generated_at,
        "source_inventory": str(args.inventory.resolve()),
        "source_inventory_sha256": sha256_file(args.inventory),
        "seed": args.seed,
        "labels": labels,
        "class_names": {str(label): class_names[label] for label in labels},
        "rules": {
            "group_unit": "road_id",
            "outer_strategy": f"{args.folds}-fold road-grouped cross-validation",
            "inner_validation": (
                "whole-road subset selected from non-test outer folds"
            ),
            "official_test_policy": (
                "official reference trajectories are withheld from CV and "
                "remain available only for separate benchmark-style evaluation"
            ),
            "primary_evaluation": (
                f"deterministic balanced subset of {args.eval_per_class} "
                "samples per class for both validation and road-unseen test"
            ),
            "supplementary_evaluation": "all assigned validation/test samples",
            "training_balance": (
                "retain all training samples; use a class-aware sampler or "
                "class weights during model training"
            ),
            "model_selection": (
                "validation macro F1 only; outer test fold evaluated after "
                "configuration lock"
            ),
            "reporting": (
                "mean and standard deviation across outer folds, per-class "
                "recall, confusion matrix, and all-sample supplementary metrics"
            ),
            "full_19_class_gate": (
                f"every included class must occur on at least {args.folds} "
                f"development roads and supply at least "
                f"{args.folds * args.eval_per_class} development instances; "
                "these are necessary rather than sufficient conditions, and "
                "the planner must still pass the per-fold coverage checks; "
                "otherwise reduce folds or declare the class non-evaluable "
                "under the strict road-domain protocol"
            ),
        },
        "input_summary": {
            "all_instance_count": len(all_instances),
            "development_instance_count": len(development),
            "official_test_withheld_count": len(official_test),
            "development_road_count": len(development_roads),
            "official_test_road_count": len(
                {str(item["road_id"]) for item in official_test}
            ),
            "development_class_histogram": histogram(development, labels),
            "official_test_class_histogram": histogram(official_test, labels),
            "road_support_by_class": {
                str(label): sum(
                    road_counts[road][label] > 0 for road in development_roads
                )
                for label in labels
            },
        },
        "outer_folds": fold_summaries,
        "runs": run_summaries,
    }
    atomic_json(protocol_path, protocol)
    plot_protocol(
        figure_path, labels, class_names, fold_summaries, run_summaries
    )

    validation = {
        "status": "passed",
        "stage": "B5c",
        "generated_at": generated_at,
        "checks": {
            "inventory_complete": True,
            "official_test_samples_excluded_from_cv": all(
                row["is_official_test"] == 0 for row in sample_assignment_rows
            ),
            "road_overlap_count_by_run": {
                str(run["run_index"]): run["road_overlap_count"]
                for run in run_summaries
            },
            "all_outer_folds_meet_eval_minimum": all(
                all(
                    fold["class_histogram"][str(label)]
                    >= args.eval_per_class
                    for label in labels
                )
                for fold in fold_summaries
            ),
            "all_balanced_eval_subsets_exact": all(
                all(
                    run["balanced_evaluation_histogram"][split][str(label)]
                    == args.eval_per_class
                    for split in ("val", "test")
                    for label in labels
                )
                for run in run_summaries
            ),
            "all_training_splits_meet_minimum": all(
                all(
                    run["class_histograms"]["train"][str(label)]
                    >= args.min_train_per_class
                    for label in labels
                )
                for run in run_summaries
            ),
            "sample_assignment_count_exact": (
                len(sample_assignment_rows)
                == len(development) * args.folds
            ),
            "each_sample_has_one_split_per_run": (
                len(
                    {
                        (row["run_index"], row["sample_key"])
                        for row in sample_assignment_rows
                    }
                )
                == len(sample_assignment_rows)
            ),
            "every_development_road_is_test_once": all(
                sum(
                    road in set(fold["roads"]) for fold in fold_summaries
                )
                == 1
                for road in development_roads
            ),
        },
        "summary": {
            "fold_count": args.folds,
            "run_count": len(run_summaries),
            "development_road_count": len(development_roads),
            "development_sample_count": len(development),
            "official_test_withheld_count": len(official_test),
            "balanced_validation_samples_per_run": (
                args.eval_per_class * len(labels)
            ),
            "balanced_test_samples_per_run": args.eval_per_class * len(labels),
            "road_support_by_class": protocol["input_summary"][
                "road_support_by_class"
            ],
        },
        "artifacts": {
            "protocol": str(protocol_path),
            "road_class_counts": str(road_counts_path),
            "road_assignments": str(road_assignments_path),
            "sample_assignments": str(sample_assignments_path),
            "figure": str(figure_path),
        },
    }
    if not all(
        value is True
        or (
            isinstance(value, dict)
            and all(item == 0 for item in value.values())
        )
        for value in validation["checks"].values()
    ):
        validation["status"] = "failed"
    atomic_json(validation_path, validation)
    if validation["status"] != "passed":
        raise ValueError("B5c validation failed")
    print(json.dumps(validation["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
