"""Plan shared assets for full benchmark and road-domain evaluation tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    build_road_class_counts,
    choose_inner_validation_roads,
    select_balanced_evaluation_keys,
    stable_sample_seed,
)


BENCHMARK_CLASS_COUNT = 19
ROAD_PROFILE_FOLDS = 3
ROAD_PROFILE_PER_CLASS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--species-csv", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--knn-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-val-per-class", type=int, default=30)
    parser.add_argument("--road-eval-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_species(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return sorted(
        [
            {
                "benchmark_label_id": int(row["benchmark_label_id"]),
                "scientific_name": row["scientific_name"],
                "abbreviation": row["abbreviation"],
            }
            for row in rows
        ],
        key=lambda item: int(item["benchmark_label_id"]),
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(text, encoding="utf-8")
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


def stable_rank(sample_key: str, namespace: str, seed: int) -> bytes:
    return hashlib.sha256(
        f"{seed}|{namespace}|{sample_key}".encode("utf-8")
    ).digest()


def histogram(
    rows: list[dict[str, object]],
    labels: list[int],
) -> dict[str, int]:
    counts = Counter(int(row["benchmark_label_id"]) for row in rows)
    return {str(label): counts[label] for label in labels}


def quantile(sorted_values: list[int], fraction: float) -> int:
    index = min(
        len(sorted_values) - 1,
        int((len(sorted_values) - 1) * fraction),
    )
    return sorted_values[index]


def reference_asset_stats(dataset_root: Path) -> dict[str, object]:
    point_sizes = sorted(
        path.stat().st_size for path in dataset_root.glob("points/**/*.npz")
    )
    image_sizes = sorted(
        path.stat().st_size for path in dataset_root.glob("images/**/*.jpg")
    )
    if not point_sizes or not image_sizes or len(point_sizes) != len(image_sizes):
        raise ValueError("Reference dataset asset samples are incomplete")

    def summary(values: list[int]) -> dict[str, object]:
        return {
            "count": len(values),
            "total_bytes": sum(values),
            "mean_bytes": statistics.mean(values),
            "median_bytes": statistics.median(values),
            "p95_bytes": quantile(values, 0.95),
            "maximum_bytes": max(values),
        }

    return {
        "reference_root": str(dataset_root.resolve()),
        "points": summary(point_sizes),
        "images": summary(image_sizes),
    }


def projected_bytes(per_sample_bytes: float, sample_count: int) -> int:
    return math.ceil(per_sample_bytes * sample_count)


def plot_plan(
    path: Path,
    benchmark_summary: dict[str, object],
    road_runs: list[dict[str, object]],
    budget: dict[str, object],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#2A6F97", "#E07A5F", "#3D8D7A"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    split_names = ["train", "val", "test"]
    split_counts = [
        int(benchmark_summary["split_counts"][split]) for split in split_names
    ]
    axes[0].bar(split_names, split_counts, color=colors)
    axes[0].set_title("19-class benchmark track")
    axes[0].set_ylabel("Samples")
    axes[0].grid(axis="y", alpha=0.2)
    for index, value in enumerate(split_counts):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")

    run_x = np.arange(len(road_runs))
    width = 0.23
    for split_index, split in enumerate(split_names):
        axes[1].bar(
            run_x + (split_index - 1) * width,
            [run["road_counts"][split] for run in road_runs],
            width,
            label=split,
            color=colors[split_index],
        )
    axes[1].set_title("16-class road-domain track")
    axes[1].set_ylabel("Whole roads")
    axes[1].set_xticks(run_x)
    axes[1].set_xticklabels([f"Run {index + 1}" for index in run_x])
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    names = ["Mean assets", "P95 assets", "P95 + 25%", "Free disk"]
    gib = 1024**3
    values = [
        budget["shared_assets_mean_bytes"] / gib,
        budget["shared_assets_p95_bytes"] / gib,
        budget["shared_assets_p95_with_safety_bytes"] / gib,
        budget["free_disk_bytes"] / gib,
    ]
    axes[2].bar(
        names,
        values,
        color=["#2A6F97", "#2A6F97", "#E07A5F", "#3D8D7A"],
    )
    axes[2].set_title("Shared-asset disk budget")
    axes[2].set_ylabel("GiB")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.2)
    for index, value in enumerate(values):
        axes[2].text(index, value, f"{value:.1f}", ha="center", va="bottom")

    figure.suptitle("C1b dual-track dataset plan", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.benchmark_val_per_class <= 0 or args.road_eval_per_class <= 0:
        raise ValueError("Evaluation quotas must be positive")
    if args.safety_factor < 1.0:
        raise ValueError("safety-factor must be at least 1.0")

    inventory = read_json(args.inventory)
    audit = read_json(args.audit)
    species = read_species(args.species_csv)
    if not isinstance(inventory, dict) or not isinstance(audit, dict):
        raise ValueError("Invalid inventory or audit")
    labels = [int(item["benchmark_label_id"]) for item in species]
    if labels != list(range(BENCHMARK_CLASS_COUNT)):
        raise ValueError("The species catalog must contain labels 0..18")
    if inventory["summary"]["completed_trajectory_count"] != inventory["trajectory_count"]:
        raise ValueError("Inventory is incomplete")

    class_names = {
        int(item["benchmark_label_id"]): str(item["scientific_name"])
        for item in species
    }
    instances = list(inventory["instances"])
    development = [item for item in instances if not item["is_official_test"]]
    official_test = [item for item in instances if item["is_official_test"]]

    benchmark_val_keys = select_balanced_evaluation_keys(
        development,
        labels,
        per_class=args.benchmark_val_per_class,
        seed=args.seed,
        namespace="benchmark-validation",
    )
    benchmark_balanced_test_keys = select_balanced_evaluation_keys(
        official_test,
        labels,
        per_class=args.benchmark_val_per_class,
        seed=args.seed,
        namespace="benchmark-balanced-test",
    )

    class_index_19 = {label: label for label in labels}
    shared_asset_rows = []
    benchmark_rows = []
    for item in sorted(instances, key=lambda row: str(row["sample_key"])):
        key = str(item["sample_key"])
        label = int(item["benchmark_label_id"])
        split = (
            "test"
            if bool(item["is_official_test"])
            else ("val" if key in benchmark_val_keys else "train")
        )
        shared_asset_rows.append(
            {
                "sample_key": key,
                "benchmark_label_id": label,
                "scientific_name": class_names[label],
                "road_id": str(item["road_id"]),
                "trajectory_id": str(item["trajectory_id"]),
                "tree_id": int(item["tree_id"]),
                "source_point_count": int(item["point_count"]),
                "sample_seed": stable_sample_seed(key, args.seed),
                "point_path": f"assets/points/{key}.npz",
                "image_path": f"assets/images/{key}.jpg",
                "is_official_test": int(bool(item["is_official_test"])),
            }
        )
        benchmark_rows.append(
            {
                "sample_key": key,
                "split": split,
                "class_index": class_index_19[label],
                "benchmark_label_id": label,
                "scientific_name": class_names[label],
                "road_id": str(item["road_id"]),
                "trajectory_id": str(item["trajectory_id"]),
                "tree_id": int(item["tree_id"]),
                "point_path": f"assets/points/{key}.npz",
                "image_path": f"assets/images/{key}.jpg",
                "balanced_evaluation_selected": int(
                    split == "val"
                    or (split == "test" and key in benchmark_balanced_test_keys)
                ),
                "is_official_test": int(bool(item["is_official_test"])),
            }
        )

    benchmark_split_rows = {
        split: [row for row in benchmark_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    benchmark_summary = {
        "class_count": len(labels),
        "sample_count": len(benchmark_rows),
        "split_counts": {
            split: len(rows) for split, rows in benchmark_split_rows.items()
        },
        "split_class_histograms": {
            split: histogram(rows, labels)
            for split, rows in benchmark_split_rows.items()
        },
        "balanced_test_count": len(benchmark_balanced_test_keys),
        "validation_policy": (
            f"deterministic {args.benchmark_val_per_class} samples per class "
            "from non-official-reference trajectories"
        ),
        "test_policy": "all official reference trajectory instances",
    }

    road_profile = next(
        (
            profile
            for profile in audit["profiles"]
            if int(profile["folds"]) == ROAD_PROFILE_FOLDS
            and int(profile["per_class"]) == ROAD_PROFILE_PER_CLASS
        ),
        None,
    )
    if road_profile is None or road_profile["joint_plan_status"] != "passed":
        raise ValueError("The audited 16-class road profile is unavailable")
    road_labels = [
        int(label) for label in road_profile["individually_feasible_labels"]
    ]
    if args.road_eval_per_class != ROAD_PROFILE_PER_CLASS:
        raise ValueError(
            "road-eval-per-class must match the audited 20-sample profile"
        )
    road_class_index = {
        label: index for index, label in enumerate(road_labels)
    }
    road_development = [
        item
        for item in development
        if int(item["benchmark_label_id"]) in set(road_labels)
    ]
    road_counts = build_road_class_counts(road_development, road_labels)

    road_assignment_rows = []
    road_sample_rows = []
    road_run_summaries = []
    all_roads = set(road_counts)
    for run_index, fold in enumerate(road_profile["outer_folds"]):
        test_roads = set(str(road) for road in fold["roads"])
        remaining_count = len(all_roads - test_roads)
        target_validation_roads = max(1, math.ceil(remaining_count * 0.2))
        val_roads = set(
            choose_inner_validation_roads(
                road_counts,
                road_labels,
                test_roads,
                validation_minimum={
                    label: args.road_eval_per_class for label in road_labels
                },
                training_minimum={
                    label: args.road_eval_per_class for label in road_labels
                },
                target_road_count=target_validation_roads,
                seed=args.seed + run_index,
                max_candidates=500_000,
            )
        )
        train_roads = all_roads - test_roads - val_roads
        split_roads = {
            "train": train_roads,
            "val": val_roads,
            "test": test_roads,
        }
        if any(
            split_roads[left] & split_roads[right]
            for left, right in (
                ("train", "val"),
                ("train", "test"),
                ("val", "test"),
            )
        ):
            raise ValueError(f"Road leakage in run {run_index}")

        split_instances = {
            split: [
                item
                for item in road_development
                if str(item["road_id"]) in roads
            ]
            for split, roads in split_roads.items()
        }
        evaluation_keys = {
            split: select_balanced_evaluation_keys(
                split_instances[split],
                road_labels,
                per_class=args.road_eval_per_class,
                seed=args.seed,
                namespace=f"road-run-{run_index}-{split}",
            )
            for split in ("val", "test")
        }

        for road in sorted(all_roads):
            split = next(
                name for name, roads in split_roads.items() if road in roads
            )
            road_assignment_rows.append(
                {
                    "run_index": run_index,
                    "road_id": road,
                    "split": split,
                    "sample_count": sum(road_counts[road].values()),
                    **{
                        f"label_{label}_count": road_counts[road][label]
                        for label in road_labels
                    },
                }
            )

        for split in ("train", "val", "test"):
            for item in split_instances[split]:
                key = str(item["sample_key"])
                label = int(item["benchmark_label_id"])
                road_sample_rows.append(
                    {
                        "run_index": run_index,
                        "sample_key": key,
                        "split": split,
                        "class_index": road_class_index[label],
                        "benchmark_label_id": label,
                        "scientific_name": class_names[label],
                        "road_id": str(item["road_id"]),
                        "trajectory_id": str(item["trajectory_id"]),
                        "tree_id": int(item["tree_id"]),
                        "point_path": f"assets/points/{key}.npz",
                        "image_path": f"assets/images/{key}.jpg",
                        "balanced_evaluation_selected": int(
                            split in evaluation_keys
                            and key in evaluation_keys[split]
                        ),
                        "is_official_test": 0,
                    }
                )

        road_run_summaries.append(
            {
                "run_index": run_index,
                "road_counts": {
                    split: len(roads) for split, roads in split_roads.items()
                },
                "roads": {
                    split: sorted(roads) for split, roads in split_roads.items()
                },
                "sample_counts": {
                    split: len(rows) for split, rows in split_instances.items()
                },
                "class_histograms": {
                    split: histogram(rows, road_labels)
                    for split, rows in split_instances.items()
                },
                "balanced_evaluation_histograms": {
                    split: {
                        str(label): sum(
                            int(item["benchmark_label_id"]) == label
                            and str(item["sample_key"]) in evaluation_keys[split]
                            for item in split_instances[split]
                        )
                        for label in road_labels
                    }
                    for split in ("val", "test")
                },
                "road_overlap_count": 0,
            }
        )

    output_dir = args.output_dir.resolve()
    shared_path = output_dir / "shared_assets.csv"
    benchmark_path = output_dir / "benchmark_19.csv"
    road_samples_path = output_dir / "road_domain_16_samples.csv"
    road_assignments_path = output_dir / "road_domain_16_roads.csv"
    classes_19_path = output_dir / "classes_19.json"
    classes_16_path = output_dir / "classes_16.json"
    plan_path = output_dir / "dual_track_plan.json"
    budget_path = output_dir / "disk_budget.json"
    budget_csv_path = output_dir / "disk_budget.csv"
    validation_path = output_dir / "validation.json"
    figure_path = output_dir / "dual_track_plan.png"

    shared_fields = [
        "sample_key",
        "benchmark_label_id",
        "scientific_name",
        "road_id",
        "trajectory_id",
        "tree_id",
        "source_point_count",
        "sample_seed",
        "point_path",
        "image_path",
        "is_official_test",
    ]
    track_fields = [
        "sample_key",
        "split",
        "class_index",
        "benchmark_label_id",
        "scientific_name",
        "road_id",
        "trajectory_id",
        "tree_id",
        "point_path",
        "image_path",
        "balanced_evaluation_selected",
        "is_official_test",
    ]
    atomic_csv(shared_path, shared_asset_rows, shared_fields)
    atomic_csv(benchmark_path, benchmark_rows, track_fields)
    atomic_csv(
        road_samples_path,
        road_sample_rows,
        ["run_index"] + track_fields,
    )
    atomic_csv(
        road_assignments_path,
        road_assignment_rows,
        ["run_index", "road_id", "split", "sample_count"]
        + [f"label_{label}_count" for label in road_labels],
    )
    atomic_json(
        classes_19_path,
        [
            {
                "class_index": class_index_19[label],
                "benchmark_label_id": label,
                "scientific_name": class_names[label],
            }
            for label in labels
        ],
    )
    atomic_json(
        classes_16_path,
        [
            {
                "class_index": road_class_index[label],
                "benchmark_label_id": label,
                "scientific_name": class_names[label],
            }
            for label in road_labels
        ],
    )

    asset_stats = reference_asset_stats(args.reference_dataset)
    asset_count = len(shared_asset_rows)
    point_mean = projected_bytes(
        float(asset_stats["points"]["mean_bytes"]), asset_count
    )
    point_p95 = projected_bytes(
        float(asset_stats["points"]["p95_bytes"]), asset_count
    )
    image_mean = projected_bytes(
        float(asset_stats["images"]["mean_bytes"]), asset_count
    )
    image_p95 = projected_bytes(
        float(asset_stats["images"]["p95_bytes"]), asset_count
    )
    shared_mean = point_mean + image_mean
    shared_p95 = point_p95 + image_p95
    shared_p95_safety = math.ceil(shared_p95 * args.safety_factor)

    knn_reference_count = int(asset_stats["points"]["count"])
    knn_all19 = projected_bytes(
        args.knn_cache.stat().st_size / knn_reference_count,
        asset_count,
    )
    knn_road16 = projected_bytes(
        args.knn_cache.stat().st_size / knn_reference_count,
        len(road_development),
    )
    feature_all19 = projected_bytes(
        args.feature_cache.stat().st_size / knn_reference_count,
        asset_count,
    )
    disk = shutil.disk_usage(output_dir.anchor)
    budget = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "shared_asset_count": asset_count,
        "reference_asset_stats": asset_stats,
        "point_assets_mean_bytes": point_mean,
        "point_assets_p95_bytes": point_p95,
        "image_assets_mean_bytes": image_mean,
        "image_assets_p95_bytes": image_p95,
        "shared_assets_mean_bytes": shared_mean,
        "shared_assets_p95_bytes": shared_p95,
        "safety_factor": args.safety_factor,
        "shared_assets_p95_with_safety_bytes": shared_p95_safety,
        "optional_knn_cache_all19_bytes": knn_all19,
        "optional_knn_cache_road16_bytes": knn_road16,
        "optional_frozen_feature_cache_all19_bytes": feature_all19,
        "free_disk_bytes": disk.free,
        "free_after_safe_shared_export_bytes": disk.free - shared_p95_safety,
        "shared_assets_written_once": True,
        "dual_track_duplicate_asset_bytes": 0,
    }
    budget_rows = [
        {"item": key, "bytes": value, "gib": value / 1024**3}
        for key, value in budget.items()
        if key.endswith("_bytes") and isinstance(value, int)
    ]
    atomic_json(budget_path, budget)
    atomic_csv(budget_csv_path, budget_rows, ["item", "bytes", "gib"])

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    plan = {
        "schema_version": 1,
        "stage": "C1b",
        "generated_at": generated_at,
        "seed": args.seed,
        "source_inventory": str(args.inventory.resolve()),
        "source_inventory_sha256": sha256_file(args.inventory),
        "source_audit": str(args.audit.resolve()),
        "source_audit_sha256": sha256_file(args.audit),
        "asset_policy": {
            "physical_store": "one shared assets/points and assets/images tree",
            "benchmark_track": "references all 17,134 shared assets",
            "road_domain_track": (
                "references the 16-class development subset of the same assets"
            ),
            "duplicate_assets_between_tracks": False,
        },
        "benchmark_19": {
            "summary": benchmark_summary,
            "class_labels": labels,
            "manifest": str(benchmark_path),
        },
        "road_domain_16": {
            "class_labels": road_labels,
            "excluded_labels": sorted(set(labels) - set(road_labels)),
            "folds": ROAD_PROFILE_FOLDS,
            "balanced_evaluation_per_class": args.road_eval_per_class,
            "sample_count_per_run": len(road_development),
            "runs": road_run_summaries,
            "sample_manifest": str(road_samples_path),
            "road_manifest": str(road_assignments_path),
        },
        "disk_budget": budget,
    }
    atomic_json(plan_path, plan)
    plot_plan(figure_path, benchmark_summary, road_run_summaries, budget)

    benchmark_keys = [str(row["sample_key"]) for row in benchmark_rows]
    shared_keys = {str(row["sample_key"]) for row in shared_asset_rows}
    shared_paths = {
        str(row["sample_key"]): (
            str(row["point_path"]),
            str(row["image_path"]),
        )
        for row in shared_asset_rows
    }
    validation = {
        "status": "passed",
        "stage": "C1b",
        "generated_at": generated_at,
        "checks": {
            "shared_asset_keys_unique": len(shared_keys) == len(shared_asset_rows),
            "benchmark_uses_every_shared_asset_once": (
                len(benchmark_keys) == len(shared_keys)
                and set(benchmark_keys) == shared_keys
            ),
            "official_test_only_in_benchmark_test": all(
                bool(row["is_official_test"]) == (row["split"] == "test")
                for row in benchmark_rows
            ),
            "benchmark_validation_exact_per_class": all(
                benchmark_summary["split_class_histograms"]["val"][str(label)]
                == args.benchmark_val_per_class
                for label in labels
            ),
            "benchmark_balanced_test_exact_per_class": all(
                sum(
                    row["split"] == "test"
                    and int(row["benchmark_label_id"]) == label
                    and bool(row["balanced_evaluation_selected"])
                    for row in benchmark_rows
                )
                == args.benchmark_val_per_class
                for label in labels
            ),
            "road_track_has_16_classes": len(road_labels) == 16,
            "road_track_excludes_official_test": all(
                not bool(row["is_official_test"]) for row in road_sample_rows
            ),
            "road_overlap_zero_every_run": all(
                run["road_overlap_count"] == 0 for run in road_run_summaries
            ),
            "road_balanced_eval_exact_every_run": all(
                all(
                    run["balanced_evaluation_histograms"][split][str(label)]
                    == args.road_eval_per_class
                    for split in ("val", "test")
                    for label in road_labels
                )
                for run in road_run_summaries
            ),
            "road_training_minimum_every_run": all(
                all(
                    run["class_histograms"]["train"][str(label)]
                    >= args.road_eval_per_class
                    for label in road_labels
                )
                for run in road_run_summaries
            ),
            "road_sample_once_per_run": (
                len(
                    {
                        (row["run_index"], row["sample_key"])
                        for row in road_sample_rows
                    }
                )
                == len(road_sample_rows)
                == len(road_development) * ROAD_PROFILE_FOLDS
            ),
            "tracks_share_asset_paths": all(
                str(row["sample_key"]) in shared_paths
                and (
                    str(row["point_path"]),
                    str(row["image_path"]),
                )
                == shared_paths[str(row["sample_key"])]
                for row in benchmark_rows + road_sample_rows
            ),
            "disk_safe_export_fits": disk.free >= shared_p95_safety,
            "assets_exported": (output_dir / "assets").exists(),
            "training_started": any(output_dir.glob("**/*.pt")),
        },
        "summary": {
            "shared_asset_count": asset_count,
            "benchmark_split_counts": benchmark_summary["split_counts"],
            "road_class_count": len(road_labels),
            "road_sample_count_per_run": len(road_development),
            "road_run_count": len(road_run_summaries),
            "shared_assets_p95_with_safety_bytes": shared_p95_safety,
            "free_disk_bytes": disk.free,
        },
        "artifacts": {
            "plan": str(plan_path),
            "shared_assets": str(shared_path),
            "benchmark_19": str(benchmark_path),
            "road_domain_16_samples": str(road_samples_path),
            "road_domain_16_roads": str(road_assignments_path),
            "classes_19": str(classes_19_path),
            "classes_16": str(classes_16_path),
            "disk_budget": str(budget_path),
            "disk_budget_csv": str(budget_csv_path),
            "figure": str(figure_path),
        },
    }
    positive_checks = {
        key: value
        for key, value in validation["checks"].items()
        if key not in {"assets_exported", "training_started"}
    }
    if not all(value is True for value in positive_checks.values()) or any(
        validation["checks"][key]
        for key in ("assets_exported", "training_started")
    ):
        validation["status"] = "failed"
    atomic_json(validation_path, validation)
    if validation["status"] != "passed":
        raise ValueError("C1b validation failed")
    print(json.dumps(validation["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
