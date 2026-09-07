#!/usr/bin/env python3
"""Create representative crop sheets for auditing assistant pseudo-label rules."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


UNKNOWN = "Unknown / 待定"


def safe_name(value: str) -> str:
    latin = value.split(" ")[0]
    return "".join(char if char.isalnum() else "_" for char in latin)


def bbox(label: dict[str, Any]) -> dict[str, float]:
    if label.get("bbox"):
        return label["bbox"]
    xs = [float(point[0]) for point in label["points"]]
    ys = [float(point[1]) for point in label["points"]]
    return {"left": min(xs), "right": max(xs), "top": min(ys), "bottom": max(ys)}


def crop_square(image: Image.Image, label: dict[str, Any], size: int) -> Image.Image:
    box = bbox(label)
    width, height = image.size
    left, right = box["left"] * width, box["right"] * width
    top, bottom = box["top"] * height, box["bottom"] * height
    side = max(right - left, bottom - top) * 1.35
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x1, y1 = max(0, cx - side / 2), max(0, cy - side / 2)
    x2, y2 = min(width, cx + side / 2), min(height, cy + side / 2)
    result = image.crop((round(x1), round(y1), round(x2), round(y2))).convert("RGB")
    result.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (28, 31, 36))
    canvas.paste(result, ((size - result.width) // 2, (size - result.height) // 2))
    return canvas


def evenly_spaced(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    return [rows[round(index * (len(rows) - 1) / (limit - 1))] for index in range(limit)]


def write_sheet(rows: list[dict[str, Any]], species: str, review_dir: Path, target: Path) -> None:
    tile, header, columns = 300, 48, 4
    selected = evenly_spaced(sorted(rows, key=lambda row: (row["stream"], row["distance"], row["frame_key"])), 24)
    rows_count = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (tile * columns, (tile + header) * rows_count), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, item in enumerate(selected):
        record_image = review_dir / "images" / f"{item['record_stem']}.jpg"
        with Image.open(record_image) as source:
            crop = crop_square(source, item["label"], tile)
        x = (index % columns) * tile
        y = (index // columns) * (tile + header)
        sheet.paste(crop, (x, y + header))
        title = f"{item['stream']} {item['frame_id']} d={item['det']:.2f} cls={item['cls']:.3f}"
        draw.text((x + 6, y + 6), title, fill=(245, 245, 245), font=font)
        draw.text((x + 6, y + 24), species, fill=(255, 205, 70), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=91, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--review-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classifier-confidence", type=float, default=0.99)
    parser.add_argument("--detector-confidence", type=float, default=0.40)
    cfg = parser.parse_args()
    review_dir = cfg.review_dir.resolve()
    output = cfg.output_dir.resolve()
    state = json.loads(cfg.review_state.resolve().read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_path in sorted((review_dir / "records").glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        state_key = f"{record['stream_id']}__{record['frame_id']}"
        if state_key in state:
            continue
        for label in record.get("labels") or []:
            if label.get("species") != UNKNOWN:
                continue
            hint = label.get("vmms_domain_classifier_suggestion") or {}
            confidence = float(hint.get("confidence") or 0.0)
            detector = float(label.get("tree_confidence") or 0.0)
            if confidence < cfg.classifier_confidence or detector < cfg.detector_confidence:
                continue
            species = str(hint.get("top1_species") or "")
            if not species:
                continue
            grouped[species].append(
                {
                    "record_stem": record_path.stem,
                    "frame_key": state_key,
                    "stream": record["stream_id"],
                    "frame_id": record["frame_id"],
                    "distance": float(record.get("camera", {}).get("route_distance_m") or 0.0),
                    "det": detector,
                    "cls": confidence,
                    "label": label,
                }
            )
    manifest = {}
    for species, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        target = output / f"{len(rows):04d}_{safe_name(species)}.jpg"
        write_sheet(rows, species, review_dir, target)
        manifest[species] = {"eligible_instances": len(rows), "sheet": str(target)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
