#!/usr/bin/env python3
"""Classify unresolved runtime polygons that predate automatic suggestions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


UNKNOWN = "Unknown / 待定"


def crop(image: Image.Image, label: dict[str, Any], size: int = 448) -> Image.Image:
    xs = [float(point[0]) for point in label["points"]]
    ys = [float(point[1]) for point in label["points"]]
    width, height = image.size
    left, right, top, bottom = min(xs) * width, max(xs) * width, min(ys) * height, max(ys) * height
    padding = max(right - left, bottom - top) * 0.18
    bounds = (
        max(0, round(left - padding)), max(0, round(top - padding)),
        min(width, round(right + padding)), min(height, round(bottom + padding)),
    )
    result = image.crop(bounds).convert("RGB")
    result.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (24, 27, 32))
    canvas.paste(result, ((size - result.width) // 2, (size - result.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-drafts", type=Path, required=True)
    parser.add_argument("--review-state", type=Path)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--domain-model", type=Path, required=True)
    parser.add_argument("--domain-summary", type=Path, required=True)
    parser.add_argument("--web-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    cfg = parser.parse_args()
    runtime = json.loads(cfg.runtime_drafts.resolve().read_text(encoding="utf-8"))
    submitted = json.loads(cfg.review_state.resolve().read_text(encoding="utf-8")) if cfg.review_state else {}
    review_images = {}
    for path in sorted((cfg.review_dir.resolve() / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        review_images[f"{record['stream_id']}__{record['frame_id']}"] = cfg.review_dir.resolve() / "images" / f"{path.stem}.jpg"
    mapping = json.loads(cfg.domain_summary.resolve().read_text(encoding="utf-8"))["class_mapping"]
    folder_to_species = {folder: species for species, folder in mapping.items()}
    jobs = []
    for origin, collection in (("draft", runtime), ("submitted", submitted)):
        for frame_key, frame in collection.items():
            for label in frame.get("labels") or []:
                if label.get("species") == UNKNOWN and not label.get("assistant_prelabel"):
                    source_path = frame.get("source_image") or review_images[frame_key]
                    with Image.open(source_path) as source:
                        image = source.convert("RGB")
                    jobs.append((origin, frame_key, label, crop(image, label)))

    domain_model = YOLO(str(cfg.domain_model.resolve()))
    web_model = YOLO(str(cfg.web_model.resolve()))
    domain_results = domain_model.predict([job[3] for job in jobs], imgsz=224, batch=16, device=0, verbose=False)
    web_results = web_model.predict([job[3] for job in jobs], imgsz=224, batch=16, device=0, verbose=False)
    rows = []
    sheet = Image.new("RGB", (448 * 2, 520 * max(1, (len(jobs) + 1) // 2)), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, ((origin, frame_key, label, image), domain, web) in enumerate(zip(jobs, domain_results, web_results)):
        domain_top = [
            {"species": folder_to_species[domain.names[int(i)]], "confidence": float(domain.probs.data[int(i)])}
            for i in domain.probs.top5
        ]
        web_top = [
            {"class": web.names[int(i)], "confidence": float(web.probs.data[int(i)])}
            for i in web.probs.top5
        ]
        rows.append({"origin": origin, "frame_key": frame_key, "label_id": label.get("label_id"), "domain_top5": domain_top, "web_top5": web_top})
        x, y = (index % 2) * 448, (index // 2) * 520
        sheet.paste(image, (x, y + 72))
        draw.text((x + 6, y + 6), frame_key, fill="white", font=font)
        draw.text((x + 6, y + 25), f"domain: {domain_top[0]['species']} {domain_top[0]['confidence']:.3f}", fill=(255, 205, 70), font=font)
        draw.text((x + 6, y + 44), f"web: {web_top[0]['class']} {web_top[0]['confidence']:.3f}", fill=(150, 210, 255), font=font)
    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sheet.save(output / "contact_sheet.jpg", quality=93, optimize=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
