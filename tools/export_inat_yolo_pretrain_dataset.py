#!/usr/bin/env python3
"""Build a weakly supervised YOLO pretraining dataset from iNaturalist images.

The downloaded iNaturalist images have species labels but no bounding boxes.
For bootstrap detection, this exporter creates one approximate full-image box
per image. The resulting model is intended only for Street View pre-boxing;
human-reviewed Street View labels should still be used for the final detector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    original_species: str
    label_species: str
    source: str


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


def iter_images(root: Path, source: str, label_map: dict[str, str]) -> list[ImageItem]:
    if not root.exists():
        return []
    items: list[ImageItem] = []
    for species_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name.casefold()):
        image_dir = species_dir / "images"
        if not image_dir.exists():
            continue
        original_species = species_dir.name
        label_species = label_map.get(original_species, original_species)
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                items.append(
                    ImageItem(
                        path=image_path.resolve(),
                        original_species=original_species,
                        label_species=label_species,
                        source=source,
                    )
                )
    return items


def cap_per_label(items: list[ImageItem], max_per_label: int, seed: int) -> list[ImageItem]:
    if max_per_label <= 0:
        return sorted(items, key=lambda item: (item.label_species.casefold(), str(item.path).casefold()))
    grouped: dict[str, list[ImageItem]] = {}
    for item in items:
        grouped.setdefault(item.label_species, []).append(item)
    rng = random.Random(seed)
    capped: list[ImageItem] = []
    for label_species, group in grouped.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        capped.extend(shuffled[:max_per_label])
    return sorted(capped, key=lambda item: (item.label_species.casefold(), str(item.path).casefold()))


def safe_stem(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("._") or "image"


def yolo_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def link_or_copy(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"
    shutil.copy2(source, destination)
    return "copy"


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "source",
        "source_image",
        "export_image",
        "export_label",
        "original_species",
        "label_species",
        "class_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export iNaturalist images as a weak YOLO pretraining dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("external_datasets/inat_max150"))
    parser.add_argument("--supplement-root", type=Path, default=Path("external_datasets/inat_parent_name_supplement"))
    parser.add_argument("--output", type=Path, default=Path("models/yolo_pretrain_inat_parent_label_v1"))
    parser.add_argument("--label-map", type=Path, default=Path("tools/tree_species_label_merge_map.csv"))
    parser.add_argument("--max-per-label", type=int, default=120)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--box-size", type=float, default=0.94, help="Weak centered box width/height, normalized.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if not (0 < args.box_size <= 1):
        raise ValueError("--box-size must be in (0, 1].")
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    label_map = read_label_map(args.label_map)
    primary = iter_images(args.data_root, "inat_max150", label_map)
    supplement = iter_images(args.supplement_root, "inat_parent_name_supplement", label_map)

    primary_original_counts: dict[str, int] = {}
    for item in primary:
        primary_original_counts[item.original_species] = primary_original_counts.get(item.original_species, 0) + 1
    combined = primary + [
        item for item in supplement if primary_original_counts.get(item.original_species, 0) == 0
    ]
    combined = cap_per_label(combined, args.max_per_label, args.seed)

    classes = sorted({item.label_species for item in combined}, key=str.casefold)
    class_to_idx = {name: index for index, name in enumerate(classes)}

    manifest_rows: list[dict] = []
    link_counts = {"hardlink": 0, "copy": 0}
    for index, item in enumerate(combined):
        split = "val" if stable_float(f"{item.label_species}|{item.path}") < args.val_fraction else "train"
        class_index = class_to_idx[item.label_species]
        stem = safe_stem(f"{index:06d}_{item.label_species}_{item.path.stem}")
        image_out = args.output / "images" / split / f"{stem}{item.path.suffix.lower()}"
        label_out = args.output / "labels" / split / f"{stem}.txt"
        link_type = link_or_copy(item.path, image_out, args.link_mode)
        link_counts[link_type] += 1
        label_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.write_text(f"{class_index} 0.500000 0.500000 {args.box_size:.6f} {args.box_size:.6f}\n", encoding="utf-8")
        manifest_rows.append({
            "split": split,
            "source": item.source,
            "source_image": str(item.path),
            "export_image": str(image_out.resolve()),
            "export_label": str(label_out.resolve()),
            "original_species": item.original_species,
            "label_species": item.label_species,
            "class_index": class_index,
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
        "output": str(args.output.resolve()),
        "images": len(manifest_rows),
        "boxes": len(manifest_rows),
        "classes": len(classes),
        "train_images": sum(1 for row in manifest_rows if row["split"] == "train"),
        "val_images": sum(1 for row in manifest_rows if row["split"] == "val"),
        "max_per_label": args.max_per_label,
        "box_size": args.box_size,
        "label_merges": len(label_map),
        "link_counts": link_counts,
        "warning": "Weak labels use one centered approximate box per image; use only for bootstrap Street View pre-boxing.",
    }
    (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
