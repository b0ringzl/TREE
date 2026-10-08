#!/usr/bin/env python3
"""Export reviewed Street View annotations to a YOLO training dataset.

The labeler stores annotations inside dataset/<species>/<tree_id> with
YOLO-format text files whose class id is local to the reviewed species.
This exporter rewrites those labels to global class ids and creates a
standard YOLO layout without modifying the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ExportItem:
    image_path: Path
    label_path: Path
    species: str
    label_species: str
    split: str


def stable_float(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def read_label_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["original_species"].strip(): row["label_species"].strip()
            for row in reader
            if row.get("original_species") and row.get("label_species")
        }


def yolo_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def safe_stem(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "image"


def parse_label_lines(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            values = [float(part) for part in parts[1:]]
        except ValueError:
            continue
        x_center, y_center, width, height = values
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            continue
        rows.append(parts)
    return rows


def iter_items(dataset_root: Path, label_map: dict[str, str], val_fraction: float) -> list[ExportItem]:
    items: list[ExportItem] = []
    if not dataset_root.exists():
        return items
    for species_dir in sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda path: path.name.casefold()):
        species = species_dir.name
        label_species = label_map.get(species, species)
        for image_path in sorted(species_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = image_path.with_suffix(".txt")
            if not parse_label_lines(label_path):
                continue
            split = "val" if stable_float(f"{label_species}|{image_path}") < val_fraction else "train"
            items.append(
                ExportItem(
                    image_path=image_path.resolve(),
                    label_path=label_path.resolve(),
                    species=species,
                    label_species=label_species,
                    split=split,
                )
            )
    return items


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "source_image",
        "source_label",
        "export_image",
        "export_label",
        "original_species",
        "label_species",
        "class_index",
        "boxes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reviewed street-view annotations to YOLO format.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("models/streetview_yolo_dataset"))
    parser.add_argument("--label-map", type=Path, default=Path("tools/tree_species_label_merge_map.csv"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true", help="Delete the output directory before exporting.")
    args = parser.parse_args()

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    label_map = read_label_map(args.label_map)
    items = iter_items(args.dataset_root, label_map, args.val_fraction)
    classes = sorted({item.label_species for item in items}, key=str.casefold)
    class_to_idx = {name: index for index, name in enumerate(classes)}

    manifest_rows: list[dict] = []
    box_count = 0
    for index, item in enumerate(items):
        class_index = class_to_idx[item.label_species]
        label_lines = parse_label_lines(item.label_path)
        if not label_lines:
            continue
        stem = safe_stem(f"{index:06d}_{item.label_species}_{item.image_path.stem}")
        image_out = args.output / "images" / item.split / f"{stem}{item.image_path.suffix.lower()}"
        label_out = args.output / "labels" / item.split / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.image_path, image_out)
        rewritten = [f"{class_index} {' '.join(parts[1:])}" for parts in label_lines]
        label_out.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        box_count += len(rewritten)
        manifest_rows.append({
            "split": item.split,
            "source_image": str(item.image_path),
            "source_label": str(item.label_path),
            "export_image": str(image_out.resolve()),
            "export_label": str(label_out.resolve()),
            "original_species": item.species,
            "label_species": item.label_species,
            "class_index": class_index,
            "boxes": len(rewritten),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output / "manifest.csv", manifest_rows)
    (args.output / "class_to_idx.json").write_text(json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")
    yaml_text = "\n".join([
        f"path: {yolo_quote(str(args.output.resolve()))}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        "names:",
        *[f"  {index}: {yolo_quote(name)}" for index, name in enumerate(classes)],
        "",
    ])
    (args.output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    report = {
        "source_dataset": str(args.dataset_root.resolve()),
        "output": str(args.output.resolve()),
        "images": len(manifest_rows),
        "boxes": box_count,
        "classes": len(classes),
        "train_images": sum(1 for row in manifest_rows if row["split"] == "train"),
        "val_images": sum(1 for row in manifest_rows if row["split"] == "val"),
        "label_merges": len(label_map),
    }
    (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
