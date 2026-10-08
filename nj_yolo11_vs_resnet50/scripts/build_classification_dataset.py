"""Materialize the frozen NJ four-class split for Ultralytics classification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_folder(index: int, name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return f"{index}_{slug}"


def main() -> None:
    args = parse_args()
    source_package = args.source_package.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = source_package / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = sorted(manifest["classes"], key=lambda item: int(item["class_index"]))
    class_names = [str(item["scientific_name"]) for item in classes]
    expected_indices = list(range(len(class_names)))
    if [int(item["class_index"]) for item in classes] != expected_indices:
        raise ValueError("Class indices are not contiguous")

    records = list(manifest["records"])
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    inventory_rows: list[dict[str, object]] = []
    materialization = Counter()
    for record in records:
        split = str(record["split"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected split: {split}")
        class_index = int(record["class_index"])
        species = str(record["scientific_name"])
        if class_names[class_index] != species:
            raise ValueError(f"Class mapping mismatch: {record['sample_key']}")
        source = source_package / str(record["image_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        split_directory = "held_out_test" if split == "test" else f"training_view/{split}"
        destination = output_dir / split_directory / class_folder(class_index, species) / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size:
                raise FileExistsError(destination)
            materialization["existing"] += 1
        else:
            os.link(source, destination)
            materialization["hardlink"] += 1
        split_counts[split][species] += 1
        image_hash = str(record["packaged_image_sha256"])
        hash_splits[image_hash].add(split)
        inventory_rows.append(
            {
                "sample_key": record["sample_key"],
                "split": split,
                "class_index": class_index,
                "scientific_name": species,
                "source_path": str(source),
                "materialized_path": str(destination),
                "sha256": image_hash,
            }
        )

    leakage = {digest: sorted(splits) for digest, splits in hash_splits.items() if len(splits) > 1}
    observed = {split: sum(counts.values()) for split, counts in split_counts.items()}
    expected = {key: int(value) for key, value in manifest["summary"]["split_sizes"].items()}
    checks = {
        "split_sizes_match": observed == expected,
        "class_order_matches_whu": class_names
        == [
            "Cinnamomum camphora",
            "Lagerstroemia indica",
            "Magnolia grandiflora",
            "Other",
        ],
        "all_classes_in_every_split": all(len(split_counts[split]) == 4 for split in expected),
        "no_exact_content_cross_split_leakage": not leakage,
    }
    if not all(checks.values()):
        raise ValueError(f"Dataset validation failed: {checks}; leakage examples={list(leakage)[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)
    summary = {
        "status": "validated",
        "task": "NJ four-class single-tree image classification",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "class_names": class_names,
        "class_folders": {
            str(index): class_folder(index, name) for index, name in enumerate(class_names)
        },
        "split_sizes": observed,
        "class_split_counts": {
            split: dict(split_counts[split]) for split in ("train", "val", "test")
        },
        "materialization": dict(materialization),
        "checks": checks,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
