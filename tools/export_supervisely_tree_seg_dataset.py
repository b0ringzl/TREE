#!/usr/bin/env python3
"""Convert a DatasetNinja/Supervisely instance-segmentation export to YOLO-seg.

DatasetNinja datasets usually unpack as Supervisely projects with ds*/img and
ds*/ann folders. This converter merges all annotated tree-like classes into one
"tree" class by default, because the labeler only needs single-tree proposals
before human target selection.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ExportedImage:
    image_path: Path
    ann_path: Path
    split: str


def stable_float(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


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


def decode_supervisely_bitmap(bitmap: dict, width: int, height: int) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(bitmap["data"]))
    tile = Image.open(BytesIO(raw)).convert("RGBA")
    tile_mask = np.array(tile)[:, :, 3] > 0
    origin_x, origin_y = bitmap.get("origin", [0, 0])
    full = np.zeros((height, width), dtype=np.uint8)
    y2 = min(height, origin_y + tile_mask.shape[0])
    x2 = min(width, origin_x + tile_mask.shape[1])
    if y2 <= origin_y or x2 <= origin_x:
        return full
    full[origin_y:y2, origin_x:x2] = tile_mask[: y2 - origin_y, : x2 - origin_x].astype(np.uint8)
    return full


def mask_to_yolo_polygons(mask: np.ndarray, width: int, height: int, epsilon: float, min_area: float) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows: list[list[float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue
        normalized: list[float] = []
        for x, y in points:
            normalized.extend([
                max(0.0, min(1.0, float(x) / width)),
                max(0.0, min(1.0, float(y) / height)),
            ])
        if len(normalized) >= 6:
            rows.append(normalized)
    return rows


def image_for_ann(ann_path: Path, image_dir: Path) -> Path | None:
    stem = ann_path.name[:-5] if ann_path.name.endswith(".json") else ann_path.stem
    direct = image_dir / stem
    if direct.exists():
        return direct
    for suffix in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    for candidate in image_dir.glob(f"{stem}.*"):
        if candidate.suffix.lower() in IMAGE_EXTS:
            return candidate
    return None


def iter_supervisely_items(root: Path, val_fraction: float, test_fraction: float) -> list[ExportedImage]:
    items: list[ExportedImage] = []
    for ann_dir in sorted(root.rglob("ann")):
        image_dir = ann_dir.parent / "img"
        if not image_dir.exists():
            continue
        for ann_path in sorted(ann_dir.glob("*.json")):
            image_path = image_for_ann(ann_path, image_dir)
            if not image_path:
                continue
            score = stable_float(str(image_path.relative_to(root)).replace("\\", "/"))
            split = "test" if score < test_fraction else "val" if score < test_fraction + val_fraction else "train"
            items.append(ExportedImage(image_path=image_path, ann_path=ann_path, split=split))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Supervisely tree masks to YOLO segmentation format.")
    parser.add_argument("--input", type=Path, required=True, help="Unpacked DatasetNinja/Supervisely project root.")
    parser.add_argument("--output", type=Path, default=Path("models/urban_tree_seg_dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--class-name", default="tree")
    parser.add_argument("--keep-classes", nargs="*", default=None, help="Optional Supervisely class names to keep.")
    parser.add_argument("--epsilon", type=float, default=0.003, help="Polygon simplification ratio.")
    parser.add_argument("--min-area", type=float, default=32.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    keep_classes = set(args.keep_classes or [])
    items = iter_supervisely_items(args.input, args.val_fraction, args.test_fraction)
    rows = []
    label_count = 0
    used_images = 0
    for index, item in enumerate(items):
        ann = json.loads(item.ann_path.read_text(encoding="utf-8"))
        width = int(ann.get("size", {}).get("width") or Image.open(item.image_path).width)
        height = int(ann.get("size", {}).get("height") or Image.open(item.image_path).height)
        yolo_lines = []
        for obj in ann.get("objects", []):
            if keep_classes and obj.get("classTitle") not in keep_classes:
                continue
            bitmap = obj.get("bitmap")
            if not bitmap or "data" not in bitmap:
                continue
            mask = decode_supervisely_bitmap(bitmap, width, height)
            for polygon in mask_to_yolo_polygons(mask, width, height, args.epsilon, args.min_area):
                yolo_lines.append("0 " + " ".join(f"{value:.6f}" for value in polygon))
        if not yolo_lines:
            continue

        stem = safe_stem(f"{index:06d}_{item.image_path.stem}")
        image_out = args.output / "images" / item.split / f"{stem}{item.image_path.suffix.lower()}"
        label_out = args.output / "labels" / item.split / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.image_path, image_out)
        label_out.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        label_count += len(yolo_lines)
        used_images += 1
        rows.append({
            "split": item.split,
            "source_image": str(item.image_path.resolve()),
            "source_annotation": str(item.ann_path.resolve()),
            "export_image": str(image_out.resolve()),
            "export_label": str(label_out.resolve()),
            "instances": len(yolo_lines),
        })

    yaml_text = "\n".join([
        f"path: {yolo_quote(str(args.output.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 1",
        "names:",
        f"  0: {yolo_quote(args.class_name)}",
        "",
    ])
    (args.output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "images": used_images,
        "instances": label_count,
        "train_images": sum(1 for row in rows if row["split"] == "train"),
        "val_images": sum(1 for row in rows if row["split"] == "val"),
        "test_images": sum(1 for row in rows if row["split"] == "test"),
        "class_name": args.class_name,
    }
    (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
