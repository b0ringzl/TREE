#!/usr/bin/env python3
"""Materialize a leakage-safe YOLO instance-segmentation dataset from crop manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--minimum-crop-pixels", type=int, default=24)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return result.strip("._") or "image"


def yolo_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def accepted_polygons(
    label_path: Path,
    width: int,
    height: int,
    padding: float,
    minimum_pixels: int,
) -> tuple[list[list[float]], Counter[str]]:
    polygons: list[list[float]] = []
    audit: Counter[str] = Counter()
    for raw in label_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = raw.strip().split()
        try:
            values = [float(value) for value in parts]
        except ValueError:
            audit["invalid_numeric"] += 1
            continue
        if len(values) == 5:
            _, cx, cy, box_width, box_height = values
            x0, y0 = cx - box_width / 2, cy - box_height / 2
            x1, y1 = cx + box_width / 2, cy + box_height / 2
            coordinates = [x0, y0, x1, y0, x1, y1, x0, y1]
            audit["box_converted_to_polygon"] += 1
        elif len(values) >= 7 and (len(values) - 1) % 2 == 0:
            coordinates = values[1:]
            xs, ys = coordinates[0::2], coordinates[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            audit["polygon"] += 1
        else:
            audit["invalid_shape"] += 1
            continue
        if any(value < 0 or value > 1 for value in coordinates):
            audit["out_of_range"] += 1
            continue
        if x1 <= x0 or y1 <= y0:
            audit["degenerate"] += 1
            continue
        box_width, box_height = x1 - x0, y1 - y0
        padded_width = min(1.0, x1 + box_width * padding) - max(0.0, x0 - box_width * padding)
        padded_height = min(1.0, y1 + box_height * padding) - max(0.0, y0 - box_height * padding)
        if padded_width * width < minimum_pixels or padded_height * height < minimum_pixels:
            audit["too_small"] += 1
            continue
        polygons.append(coordinates)
    return polygons, audit


def materialize_image(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Crop manifest is empty")

    class_names_by_index: dict[int, str] = {}
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        class_index = int(row["class_index"])
        species = row["species"]
        if class_index in class_names_by_index and class_names_by_index[class_index] != species:
            raise ValueError(f"Class-index conflict: {class_index}")
        class_names_by_index[class_index] = species
        grouped[row["image_path"]].append(row)
    class_names = [class_names_by_index[index] for index in range(len(class_names_by_index))]

    inventory: list[dict[str, object]] = []
    audit: Counter[str] = Counter()
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    hash_splits: defaultdict[str, set[str]] = defaultdict(set)
    split_images: Counter[str] = Counter()
    split_instances: Counter[str] = Counter()
    per_class_instances: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for image_index, (relative_image, image_rows) in enumerate(sorted(grouped.items())):
        species_values = {row["species"] for row in image_rows}
        split_values = {row["split"] for row in image_rows}
        class_values = {int(row["class_index"]) for row in image_rows}
        if len(species_values) != 1 or len(split_values) != 1 or len(class_values) != 1:
            raise ValueError(f"Inconsistent manifest rows for {relative_image}")
        species, split, class_index = species_values.pop(), split_values.pop(), class_values.pop()
        source = (PROJECT_ROOT / relative_image).resolve()
        source_label = source.with_suffix(".txt")
        if not source.is_file() or not source_label.is_file():
            raise FileNotFoundError(source if not source.is_file() else source_label)
        with Image.open(source) as image:
            width, height = image.size
        polygons, image_audit = accepted_polygons(
            source_label, width, height, args.crop_padding, args.minimum_crop_pixels
        )
        audit.update(image_audit)
        if len(polygons) != len(image_rows):
            raise ValueError(
                f"Annotation count mismatch for {relative_image}: labels={len(polygons)} manifest={len(image_rows)}"
            )

        digest = hashlib.sha256(relative_image.encode("utf-8")).hexdigest()[:16]
        stem = safe_name(f"{image_index:05d}_{digest}_{source.stem}")
        if split == "test":
            image_out = output_dir / "held_out_test" / "images" / f"{stem}{source.suffix.lower()}"
            label_out = output_dir / "held_out_test" / "labels" / f"{stem}.txt"
        else:
            image_out = output_dir / "training_view" / "images" / split / f"{stem}{source.suffix.lower()}"
            label_out = output_dir / "training_view" / "labels" / split / f"{stem}.txt"
        method = materialize_image(source, image_out)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.write_text(
            "\n".join(
                f"{class_index} " + " ".join(f"{value:.8f}" for value in polygon)
                for polygon in polygons
            )
            + "\n",
            encoding="utf-8",
        )
        source_sha = image_rows[0]["sha256"]
        group_id = image_rows[0]["group_id"]
        group_splits[group_id].add(split)
        hash_splits[source_sha].add(split)
        split_images[split] += 1
        split_instances[split] += len(polygons)
        per_class_instances[species][split] += len(polygons)
        audit[method] += 1
        inventory.append(
            {
                "split": split,
                "species": species,
                "class_index": class_index,
                "group_id": group_id,
                "source_image": str(source),
                "source_label": str(source_label),
                "export_image": str(image_out),
                "export_label": str(label_out),
                "instances": len(polygons),
                "source_sha256": source_sha,
            }
        )

    leaking_groups = [key for key, values in group_splits.items() if len(values) > 1]
    leaking_hashes = [key for key, values in hash_splits.items() if len(values) > 1]
    if leaking_groups or leaking_hashes:
        raise ValueError(f"Split leakage: groups={len(leaking_groups)} hashes={len(leaking_hashes)}")

    with (output_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    names = [f"  {index}: {yolo_quote(name)}" for index, name in enumerate(class_names)]
    training_yaml = [
        f"path: {yolo_quote(str((output_dir / 'training_view').resolve()))}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
        *names,
        "",
    ]
    test_yaml = [
        f"path: {yolo_quote(str(output_dir.resolve()))}",
        "train: training_view/images/train",
        "val: training_view/images/val",
        "test: held_out_test/images",
        f"nc: {len(class_names)}",
        "names:",
        *names,
        "",
    ]
    (output_dir / "data.yaml").write_text("\n".join(training_yaml), encoding="utf-8")
    (output_dir / "test.yaml").write_text("\n".join(test_yaml), encoding="utf-8")
    summary = {
        "status": "validated",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "classes": len(class_names),
        "class_names": class_names,
        "images": len(inventory),
        "instances": len(rows),
        "split_images": dict(split_images),
        "split_instances": dict(split_instances),
        "per_class_instances": {name: dict(per_class_instances[name]) for name in class_names},
        "annotation_audit": dict(audit),
        "test_hidden_from_training_yaml": True,
        "group_leakage": 0,
        "content_hash_leakage": 0,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
