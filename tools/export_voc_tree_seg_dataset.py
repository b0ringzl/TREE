#!/usr/bin/env python3
"""Convert a VOC-style tree semantic-segmentation dataset to YOLO-seg.

The Urban Street: Tree download may unpack as VOCdevkit/VOC2012 with
JPEGImages, SegmentationClass, and ImageSets/Segmentation split files. Those
masks are semantic class masks, so this exporter turns each non-background
connected tree region into a YOLO segmentation polygon. By default all tree
species are merged into one "tree" class for proposal generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
IGNORE_VALUES = {0, 255}


@dataclass(frozen=True)
class VocItem:
    image_id: str
    image_path: Path
    mask_path: Path
    split: str


def yolo_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def safe_stem(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("._") or "image"


def read_class_names(path: Path) -> list[str]:
    if not path.exists():
        return ["_background_", "tree"]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def find_image(jpeg_dir: Path, image_id: str) -> Path | None:
    for suffix in IMAGE_EXTS:
        candidate = jpeg_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def read_split_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_items(voc_root: Path, splits: list[str]) -> list[VocItem]:
    jpeg_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"
    split_dir = voc_root / "ImageSets" / "Segmentation"
    items: list[VocItem] = []
    seen: set[tuple[str, str]] = set()
    for split in splits:
        for image_id in read_split_ids(split_dir / f"{split}.txt"):
            key = (split, image_id)
            if key in seen:
                continue
            seen.add(key)
            image_path = find_image(jpeg_dir, image_id)
            mask_path = mask_dir / f"{image_id}.png"
            if image_path and mask_path.exists():
                items.append(VocItem(image_id=image_id, image_path=image_path, mask_path=mask_path, split=split))
    return items


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


def contour_to_line(class_id: int, contour: np.ndarray, width: int, height: int, epsilon: float) -> str | None:
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
    points = approx.reshape(-1, 2)
    if len(points) < 3:
        return None
    values: list[str] = [str(class_id)]
    for x, y in points:
        values.append(f"{max(0.0, min(1.0, float(x) / width)):.6f}")
        values.append(f"{max(0.0, min(1.0, float(y) / height)):.6f}")
    return " ".join(values) if len(values) >= 7 else None


def mask_to_yolo_lines(
    mask: np.ndarray,
    class_values: list[int],
    single_class: bool,
    epsilon: float,
    min_area: float,
    close_kernel: int,
    mask_scale: float,
) -> list[str]:
    if 0 < mask_scale < 1:
        mask = cv2.resize(mask, None, fx=mask_scale, fy=mask_scale, interpolation=cv2.INTER_NEAREST)
    height, width = mask.shape[:2]
    lines: list[str] = []
    kernel = None
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    for class_value in class_values:
        binary = (mask == class_value).astype(np.uint8)
        if kernel is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        class_id = 0 if single_class else class_value - 1
        for contour in contours:
            if float(cv2.contourArea(contour)) < min_area:
                continue
            line = contour_to_line(class_id, contour, width, height, epsilon)
            if line:
                lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Export VOC tree masks to YOLO segmentation format.")
    parser.add_argument("--input", type=Path, default=Path("dd/VOCdevkit/VOC2012"), help="VOC2012 root directory.")
    parser.add_argument("--output", type=Path, default=Path("models/urban_tree_voc_seg_dataset"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--class-name", default="tree")
    parser.add_argument("--single-class", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epsilon", type=float, default=0.0025, help="Polygon simplification ratio.")
    parser.add_argument("--min-area", type=float, default=256.0)
    parser.add_argument("--close-kernel", type=int, default=0, help="Optional morphology close kernel size.")
    parser.add_argument("--mask-scale", type=float, default=0.25, help="Downscale masks before contour extraction.")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test image limit.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    voc_root = args.input.resolve()
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(voc_root / "class_names.txt")
    items = iter_items(voc_root, args.splits)
    if args.limit > 0:
        items = items[: args.limit]

    if args.single_class:
        names = [args.class_name]
    else:
        names = class_names[1:]

    manifest_rows: list[dict] = []
    link_counts = {"hardlink": 0, "copy": 0}
    instance_count = 0
    skipped = 0
    for index, item in enumerate(items):
        stem = safe_stem(item.image_id)
        image_out = args.output / "images" / item.split / f"{stem}{item.image_path.suffix.lower()}"
        label_out = args.output / "labels" / item.split / f"{stem}.txt"
        if args.resume and image_out.exists() and label_out.exists():
            existing_lines = [line for line in label_out.read_text(encoding="utf-8").splitlines() if line.strip()]
            if existing_lines:
                instance_count += len(existing_lines)
                manifest_rows.append({
                    "split": item.split,
                    "image_id": item.image_id,
                    "source_image": str(item.image_path.resolve()),
                    "source_mask": str(item.mask_path.resolve()),
                    "export_image": str(image_out.resolve()),
                    "export_label": str(label_out.resolve()),
                    "class_values": "",
                    "instances": len(existing_lines),
                })
                continue

        mask = cv2.imread(str(item.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped += 1
            continue
        class_values = sorted(int(value) for value in np.unique(mask).tolist() if int(value) not in IGNORE_VALUES)
        lines = mask_to_yolo_lines(
            mask,
            class_values,
            args.single_class,
            args.epsilon,
            args.min_area * max(0.0001, args.mask_scale * args.mask_scale),
            args.close_kernel,
            args.mask_scale,
        )
        if not lines:
            skipped += 1
            continue

        link_type = link_or_copy(item.image_path, image_out, args.link_mode)
        link_counts[link_type] += 1
        label_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        instance_count += len(lines)
        manifest_rows.append({
            "split": item.split,
            "image_id": item.image_id,
            "source_image": str(item.image_path.resolve()),
            "source_mask": str(item.mask_path.resolve()),
            "export_image": str(image_out.resolve()),
            "export_label": str(label_out.resolve()),
            "class_values": " ".join(str(value) for value in class_values),
            "instances": len(lines),
        })

    fields = [
        "split",
        "image_id",
        "source_image",
        "source_mask",
        "export_image",
        "export_label",
        "class_values",
        "instances",
    ]
    with (args.output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    yaml_lines = [
        f"path: {yolo_quote(str(args.output.resolve()))}",
        "train: images/train",
        "val: images/val",
    ]
    if (args.output / "images" / "test").exists():
        yaml_lines.append("test: images/test")
    yaml_lines.extend([
        f"nc: {len(names)}",
        "names:",
        *[f"  {index}: {yolo_quote(name)}" for index, name in enumerate(names)],
        "",
    ])
    (args.output / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
    (args.output / "class_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "source": str(voc_root),
        "output": str(args.output.resolve()),
        "images": len(manifest_rows),
        "instances": instance_count,
        "skipped_images": skipped,
        "single_class": args.single_class,
        "names": names,
        "train_images": sum(1 for row in manifest_rows if row["split"] == "train"),
        "val_images": sum(1 for row in manifest_rows if row["split"] == "val"),
        "test_images": sum(1 for row in manifest_rows if row["split"] == "test"),
        "link_counts": link_counts,
    }
    (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
