"""Create the deterministic 600-tree B1 train/validation/test selection plan."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import choose_validation_roads, select_balanced_records  # noqa: E402


LABELS = (1, 2, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--species-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=140)
    parser.add_argument("--val-per-class", type=int, default=30)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def read_class_names(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = {
        int(row["benchmark_label_id"]): row["scientific_name"] for row in rows
    }
    missing = set(LABELS) - set(names)
    if missing:
        raise ValueError(f"Species CSV is missing labels: {sorted(missing)}")
    return names


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_key",
        "split",
        "class_index",
        "benchmark_label_id",
        "scientific_name",
        "road_id",
        "trajectory_id",
        "tree_id",
        "point_count",
        "sample_seed",
        "annotation_source",
        "is_official_test",
    ]
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    requested = (args.train_per_class, args.val_per_class, args.test_per_class)
    if any(count <= 0 for count in requested):
        raise ValueError("Every split quota must be positive")

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    instances = inventory["instances"]
    if inventory["summary"]["completed_trajectory_count"] != inventory["trajectory_count"]:
        raise ValueError("Inventory is incomplete")
    class_names = read_class_names(args.species_csv)
    official_test_roads = sorted(
        {str(item["road_id"]) for item in instances if item["is_official_test"]}
    )
    validation_minimum = {label: args.val_per_class for label in LABELS}
    validation_roads = choose_validation_roads(
        instances,
        LABELS,
        validation_minimum,
        excluded_roads=official_test_roads,
    )
    quotas = {
        "train": {label: args.train_per_class for label in LABELS},
        "val": {label: args.val_per_class for label in LABELS},
        "test": {label: args.test_per_class for label in LABELS},
    }
    selected, summary = select_balanced_records(
        instances,
        LABELS,
        class_names,
        validation_roads,
        quotas,
        seed=args.seed,
    )

    plan = {
        "format_version": 1,
        "source_inventory": str(args.inventory),
        "seed": args.seed,
        "target_labels": list(LABELS),
        "class_names": {str(label): class_names[label] for label in LABELS},
        "class_index_to_benchmark_label": {
            str(index): label for index, label in enumerate(LABELS)
        },
        "split_policy": {
            "test": "official WHU-STree reference trajectories",
            "validation": "whole non-test roads disjoint from official test roads",
            "train": "remaining non-test trajectories",
            "within_pool_sampling": "deterministic SHA-256 rank",
        },
        "official_test_roads": official_test_roads,
        "quotas": {
            split: {str(label): count for label, count in values.items()}
            for split, values in quotas.items()
        },
        "summary": summary,
        "records": selected,
    }
    atomic_json(args.output_json, plan)
    atomic_csv(args.output_csv, selected)
    print(json.dumps({"output": str(args.output_json), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
