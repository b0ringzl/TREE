#!/usr/bin/env python3
"""Create a route-block-separated classification dataset from VMMS human polygons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exhaustive_species_drafts import load_human_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-instances", type=int, default=8)
    return parser.parse_args()


def latin_slug(species: str) -> str:
    latin = species.split("(", 1)[0].split()
    latin = "_".join(latin[:2])
    return re.sub(r"[^A-Za-z0-9_-]+", "_", latin).strip("_")


def link_or_save(crop: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(target, quality=94, optimize=True)


def main() -> None:
    cfg = parse_args()
    review, output = cfg.review_dir.resolve(), cfg.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records = load_human_records()
    counts = Counter(
        label.get("species", "")
        for record, _ in records.values()
        for label in (record.get("labels") or [])
        if label.get("species") and not label.get("species", "").startswith("Unknown")
    )
    selected = {name for name, count in counts.items() if count >= cfg.min_instances}
    class_names = {name: f"{index:02d}_{latin_slug(name)}" for index, name in enumerate(sorted(selected))}

    blocks: dict[str, Counter[str]] = defaultdict(Counter)
    for (route, stream, frame_id), (record, _) in records.items():
        block = record.get("route_block_id") or f"{route}_{stream}"
        for label in record.get("labels") or []:
            if label.get("species") in selected:
                blocks[label["species"]][block] += 1
    val_blocks: dict[str, set[str]] = {}
    for species, block_counts in blocks.items():
        ordered = sorted(block_counts, key=lambda b: (block_counts[b], b))
        target = max(1, round(sum(block_counts.values()) * 0.20))
        chosen: set[str] = set()
        total = 0
        # Whole blocks only; keep at least one block for training.
        for block in ordered[:-1]:
            if total >= target:
                break
            chosen.add(block)
            total += block_counts[block]
        val_blocks[species] = chosen

    rows = []
    split_counts = Counter()
    for (route, stream, frame_id), (record, source_record) in sorted(records.items()):
        key = f"{route}__{stream}__{frame_id}"
        image_path = review / "images" / f"{key}.jpg"
        if not image_path.is_file():
            continue
        block = record.get("route_block_id") or f"{route}_{stream}"
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        width, height = image.size
        for index, label in enumerate(record.get("labels") or []):
            species = label.get("species", "")
            if species not in selected:
                continue
            points = label["points"]
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            x1, x2, y1, y2 = min(xs) * width, max(xs) * width, min(ys) * height, max(ys) * height
            pad = max(x2 - x1, y2 - y1) * 0.10
            box = (
                max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                min(width, int(x2 + pad + 1)), min(height, int(y2 + pad + 1)),
            )
            if box[2] - box[0] < 12 or box[3] - box[1] < 12:
                continue
            split = "val" if block in val_blocks[species] else "train"
            filename = f"{key}__{index:02d}.jpg"
            target = output / split / class_names[species] / filename
            link_or_save(image.crop(box), target)
            rows.append({
                "split": split, "class_folder": class_names[species], "species": species,
                "route": route, "stream_id": stream, "frame_id": frame_id,
                "route_block_id": block, "crop": str(target), "source_record": str(source_record),
            })
            split_counts[(split, species)] += 1

    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "ready", "min_instances": cfg.min_instances,
        "classes": len(class_names), "crops": len(rows),
        "train": sum(v for (split, _), v in split_counts.items() if split == "train"),
        "val": sum(v for (split, _), v in split_counts.items() if split == "val"),
        "class_mapping": class_names,
        "validation_blocks": {k: sorted(v) for k, v in val_blocks.items()},
        "split_policy": "species-stratified whole route blocks; no frame from a validation block appears in training for that species",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "classes", "crops", "train", "val")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
