"""Audit full-class WHU inventory coverage for road-grouped evaluation."""

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
    grouped_counts_can_meet_minimum,
    maximum_grouped_minimum,
    plan_outer_road_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--species-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--eval-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--restarts", type=int, default=3000)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_species(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        result.append(
            {
                "benchmark_label_id": int(row["benchmark_label_id"]),
                "scientific_name": row["scientific_name"],
                "abbreviation": row["abbreviation"],
                "published_instance_count": int(row["published_instance_count"]),
            }
        )
    return sorted(result, key=lambda item: int(item["benchmark_label_id"]))


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


def class_histogram(
    instances: list[dict[str, object]], labels: list[int]
) -> dict[str, int]:
    counts = Counter(int(item["benchmark_label_id"]) for item in instances)
    return {str(label): counts[label] for label in labels}


def attempt_profile(
    road_counts: dict[str, Counter[int]],
    labels: list[int],
    *,
    folds: int,
    per_class: int,
    seed: int,
    restarts: int,
) -> dict[str, object]:
    eligible = [
        label
        for label in labels
        if sum(counts[label] > 0 for counts in road_counts.values()) >= folds
        and sum(counts[label] for counts in road_counts.values())
        >= folds * per_class
    ]
    excluded = [label for label in labels if label not in eligible]
    individually_feasible = [
        label
        for label in eligible
        if grouped_counts_can_meet_minimum(
            [counts[label] for counts in road_counts.values()],
            n_splits=folds,
            minimum=per_class,
        )
    ]
    individually_infeasible = [
        label for label in labels if label not in individually_feasible
    ]
    result: dict[str, object] = {
        "folds": folds,
        "per_class": per_class,
        "necessary_eligible_labels": eligible,
        "necessary_excluded_labels": excluded,
        "individually_feasible_labels": individually_feasible,
        "individually_infeasible_labels": individually_infeasible,
        "joint_plan_status": "not_attempted",
        "joint_plan_error": None,
        "outer_folds": [],
    }
    if not individually_feasible:
        result["joint_plan_status"] = "infeasible"
        result["joint_plan_error"] = (
            "No class satisfies the independent whole-road partition check"
        )
        return result

    filtered_counts = {
        road: Counter(
            {label: counts[label] for label in individually_feasible}
        )
        for road, counts in road_counts.items()
    }
    try:
        assignment = plan_outer_road_folds(
            filtered_counts,
            individually_feasible,
            n_splits=folds,
            minimum_per_class={
                label: per_class for label in individually_feasible
            },
            seed=seed,
            restarts=restarts,
        )
    except ValueError as error:
        result["joint_plan_status"] = "infeasible"
        result["joint_plan_error"] = str(error)
        return result

    fold_rows = []
    for fold in range(folds):
        roads = sorted(
            road for road, assigned in assignment.items() if assigned == fold
        )
        counts = Counter()
        for road in roads:
            counts.update(filtered_counts[road])
        fold_rows.append(
            {
                "fold": fold,
                "roads": roads,
                "road_count": len(roads),
                "class_histogram": {
                    str(label): counts[label]
                    for label in individually_feasible
                },
                "minimum_class_count": min(
                    counts[label] for label in individually_feasible
                ),
            }
        )
    result["joint_plan_status"] = "passed"
    result["outer_folds"] = fold_rows
    return result


def plot_coverage(
    path: Path,
    rows: list[dict[str, object]],
    profiles: list[dict[str, object]],
    folds: int,
    eval_per_class: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(row["abbreviation"]) for row in rows]
    x = np.arange(len(rows))
    development = [int(row["development_instance_count"]) for row in rows]
    official = [int(row["official_test_instance_count"]) for row in rows]
    roads = [int(row["development_road_support"]) for row in rows]

    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axes[0].bar(x, development, color="#2A6F97", label="Development")
    axes[0].bar(
        x,
        official,
        bottom=development,
        color="#E07A5F",
        label="Official reference test",
    )
    axes[0].axhline(
        folds * eval_per_class,
        color="#B23A48",
        linestyle="--",
        linewidth=1.5,
        label=f"{folds} x {eval_per_class} necessary sample threshold",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Instances (log scale)")
    axes[0].set_title("Full 19-class inventory")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, roads, color="#3D8D7A")
    axes[1].axhline(
        folds,
        color="#B23A48",
        linestyle="--",
        linewidth=1.5,
        label=f"{folds}-road necessary threshold",
    )
    axes[1].set_ylabel("Development roads")
    axes[1].set_title("Road support by class")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    matrix = np.asarray(
        [
            [
                int(row["benchmark_label_id"])
                in set(profile["individually_feasible_labels"])
                for row in rows
            ]
            for profile in profiles
        ],
        dtype=np.int8,
    )
    axes[2].imshow(
        matrix,
        aspect="auto",
        cmap=matplotlib.colors.ListedColormap(["#D6D6D6", "#3D8D7A"]),
        vmin=0,
        vmax=1,
    )
    axes[2].set_yticks(np.arange(len(profiles)))
    axes[2].set_yticklabels(
        [
            f"{profile['folds']} folds, {profile['per_class']}/class"
            for profile in profiles
        ]
    )
    axes[2].set_title(
        "Exact per-class road-partition feasibility (green = feasible)"
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")

    figure.suptitle("C1a road-domain feasibility audit", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    inventory = read_json(args.inventory)
    species = read_species(args.species_csv)
    if not isinstance(inventory, dict):
        raise ValueError("Invalid inventory JSON")

    labels = [int(item["benchmark_label_id"]) for item in species]
    if labels != list(range(19)):
        raise ValueError(f"Expected benchmark labels 0..18, got {labels}")
    if inventory.get("target_benchmark_labels") != labels:
        raise ValueError("Inventory target labels do not match the 19-class catalog")
    if inventory["summary"]["completed_trajectory_count"] != inventory["trajectory_count"]:
        raise ValueError("Inventory is incomplete")

    instances = [
        item
        for item in inventory["instances"]
        if int(item["benchmark_label_id"]) in set(labels)
    ]
    development = [item for item in instances if not item["is_official_test"]]
    official = [item for item in instances if item["is_official_test"]]
    road_counts = build_road_class_counts(development, labels)
    all_counts = Counter(int(item["benchmark_label_id"]) for item in instances)
    development_counts = Counter(
        int(item["benchmark_label_id"]) for item in development
    )
    official_counts = Counter(
        int(item["benchmark_label_id"]) for item in official
    )

    class_rows = []
    for item in species:
        label = int(item["benchmark_label_id"])
        road_support = sum(
            counts[label] > 0 for counts in road_counts.values()
        )
        official_road_support = len(
            {
                str(instance["road_id"])
                for instance in official
                if int(instance["benchmark_label_id"]) == label
            }
        )
        class_rows.append(
            {
                **item,
                "inventory_instance_count": all_counts[label],
                "development_instance_count": development_counts[label],
                "official_test_instance_count": official_counts[label],
                "development_road_support": road_support,
                "official_test_road_support": official_road_support,
                "sample_threshold": args.folds * args.eval_per_class,
                "road_threshold": args.folds,
                "sample_threshold_pass": int(
                    development_counts[label]
                    >= args.folds * args.eval_per_class
                ),
                "road_threshold_pass": int(road_support >= args.folds),
                "strict_profile_necessary_pass": int(
                    development_counts[label]
                    >= args.folds * args.eval_per_class
                    and road_support >= args.folds
                ),
                "independent_per_fold_capacity_capped_at_strict_target": (
                    maximum_grouped_minimum(
                    [counts[label] for counts in road_counts.values()],
                    n_splits=args.folds,
                        upper_bound=args.eval_per_class,
                    )
                ),
                "strict_profile_individual_partition_pass": int(
                    grouped_counts_can_meet_minimum(
                        [
                            counts[label]
                            for counts in road_counts.values()
                        ],
                        n_splits=args.folds,
                        minimum=args.eval_per_class,
                    )
                ),
            }
        )

    profiles = [
        attempt_profile(
            road_counts,
            labels,
            folds=folds,
            per_class=per_class,
            seed=args.seed + index,
            restarts=args.restarts,
        )
        for index, (folds, per_class) in enumerate(
            (
                (args.folds, args.eval_per_class),
                (3, 20),
                (3, 10),
                (3, 5),
                (2, 30),
                (2, 10),
            )
        )
    ]

    road_rows = []
    for road in sorted(road_counts):
        row: dict[str, object] = {
            "road_id": road,
            "sample_count": sum(road_counts[road].values()),
        }
        for label in labels:
            row[f"label_{label}_count"] = road_counts[road][label]
        road_rows.append(row)

    output_dir = args.output_dir.resolve()
    class_path = output_dir / "class_road_coverage.csv"
    road_path = output_dir / "road_class_matrix.csv"
    audit_path = output_dir / "c1a_audit.json"
    validation_path = output_dir / "validation.json"
    figure_path = output_dir / "class_road_coverage.png"

    atomic_csv(
        class_path,
        class_rows,
        [
            "benchmark_label_id",
            "scientific_name",
            "abbreviation",
            "published_instance_count",
            "inventory_instance_count",
            "development_instance_count",
            "official_test_instance_count",
            "development_road_support",
            "official_test_road_support",
            "sample_threshold",
            "road_threshold",
            "sample_threshold_pass",
            "road_threshold_pass",
            "strict_profile_necessary_pass",
            "independent_per_fold_capacity_capped_at_strict_target",
            "strict_profile_individual_partition_pass",
        ],
    )
    atomic_csv(
        road_path,
        road_rows,
        ["road_id", "sample_count"]
        + [f"label_{label}_count" for label in labels],
    )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    strict_profile = profiles[0]
    audit = {
        "schema_version": 1,
        "stage": "C1a",
        "generated_at": generated_at,
        "source_inventory": str(args.inventory.resolve()),
        "source_inventory_sha256": sha256_file(args.inventory),
        "species_catalog": str(args.species_csv.resolve()),
        "species_catalog_sha256": sha256_file(args.species_csv),
        "policy": {
            "development_pool": "all non-official-reference trajectories",
            "official_test_policy": (
                "withheld from road-grouped planning and retained for separate "
                "benchmark-style evaluation"
            ),
            "strict_profile": {
                "folds": args.folds,
                "balanced_evaluation_per_class": args.eval_per_class,
                "necessary_sample_minimum": args.folds * args.eval_per_class,
                "necessary_road_minimum": args.folds,
                "note": (
                    "Necessary conditions do not guarantee a joint feasible "
                    "road assignment; the shared planner must also pass."
                ),
            },
        },
        "summary": {
            "trajectory_count": int(inventory["trajectory_count"]),
            "instance_count": len(instances),
            "development_instance_count": len(development),
            "official_test_instance_count": len(official),
            "development_road_count": len(road_counts),
            "official_test_road_count": len(
                {str(item["road_id"]) for item in official}
            ),
            "class_count": len(labels),
            "inventory_class_histogram": class_histogram(instances, labels),
            "development_class_histogram": class_histogram(development, labels),
            "official_test_class_histogram": class_histogram(official, labels),
            "strict_profile_necessary_eligible_count": len(
                strict_profile["necessary_eligible_labels"]
            ),
            "strict_profile_individually_feasible_count": len(
                strict_profile["individually_feasible_labels"]
            ),
            "strict_profile_joint_plan_status": strict_profile[
                "joint_plan_status"
            ],
        },
        "classes": class_rows,
        "profiles": profiles,
    }
    atomic_json(audit_path, audit)
    plot_coverage(
        figure_path,
        class_rows,
        profiles,
        args.folds,
        args.eval_per_class,
    )

    keys = [str(item["sample_key"]) for item in instances]
    published_match = all(
        int(row["inventory_instance_count"])
        == int(row["published_instance_count"])
        for row in class_rows
    )
    validation = {
        "status": "passed",
        "stage": "C1a",
        "generated_at": generated_at,
        "checks": {
            "inventory_complete": True,
            "all_19_classes_present": set(all_counts) == set(labels),
            "sample_keys_unique": len(keys) == len(set(keys)),
            "class_counts_match_published_catalog": published_match,
            "development_plus_official_equals_total": all(
                development_counts[label] + official_counts[label]
                == all_counts[label]
                for label in labels
            ),
            "original_payload_exported": False,
            "training_started": False,
        },
        "summary": audit["summary"],
        "artifacts": {
            "inventory": str(args.inventory.resolve()),
            "audit": str(audit_path),
            "class_road_coverage": str(class_path),
            "road_class_matrix": str(road_path),
            "figure": str(figure_path),
        },
    }
    if not all(
        value is True
        for key, value in validation["checks"].items()
        if key not in {"original_payload_exported", "training_started"}
    ) or any(
        validation["checks"][key]
        for key in ("original_payload_exported", "training_started")
    ):
        validation["status"] = "failed"
    atomic_json(validation_path, validation)
    if validation["status"] != "passed":
        raise ValueError("C1a validation failed")
    print(json.dumps(validation["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
