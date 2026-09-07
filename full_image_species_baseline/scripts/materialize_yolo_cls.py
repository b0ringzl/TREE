#!/usr/bin/env python3
"""Materialize a manifest as an Ultralytics classification dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_class(index: int, species: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", species).strip("_")
    return f"{index:03d}_{slug}"


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest is empty")
    mapping: dict[int, str] = {}
    for row in rows:
        index = int(row["class_index"])
        species = row["species"]
        if index in mapping and mapping[index] != species:
            raise ValueError(f"Class index {index} has multiple names")
        mapping[index] = species
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Class indices are not contiguous")

    inventory: list[dict[str, object]] = []
    methods = {"hardlink": 0, "copy": 0, "crop": 0, "existing": 0}
    for row in rows:
        split = row["split"]
        split_dir = "held_out_test" if split == "test" else f"training_view/{split}"
        class_index = int(row["class_index"])
        class_dir = safe_class(class_index, row["species"])
        source = PROJECT_ROOT / row["image_path"]
        suffix = source.suffix.lower() if row["sample_type"] == "full" else ".jpg"
        destination = output_dir / split_dir / class_dir / f"{row['sample_id']}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            methods["existing"] += 1
        elif row["sample_type"] == "full":
            try:
                os.link(source, destination)
                methods["hardlink"] += 1
            except OSError:
                shutil.copy2(source, destination)
                methods["copy"] += 1
        else:
            with Image.open(source) as image:
                image = image.convert("RGB")
                width, height = image.size
                box = (
                    max(0, math.floor(float(row["x0"]) * width)),
                    max(0, math.floor(float(row["y0"]) * height)),
                    min(width, math.ceil(float(row["x1"]) * width)),
                    min(height, math.ceil(float(row["y1"]) * height)),
                )
                image.crop(box).save(destination, format="JPEG", quality=95, subsampling=0)
            methods["crop"] += 1
        inventory.append(
            {
                "sample_id": row["sample_id"],
                "split": split,
                "class_index": class_index,
                "species": row["species"],
                "group_id": row["group_id"],
                "source_image": str(source),
                "materialized_path": str(destination),
                "source_sha256": row["sha256"],
            }
        )

    with (output_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    summary = {
        "status": "validated",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "class_names": [mapping[index] for index in range(len(mapping))],
        "class_folders": {str(index): safe_class(index, mapping[index]) for index in range(len(mapping))},
        "split_sizes": {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val", "test")},
        "materialization": methods,
        "test_hidden_from_training_view": True,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
