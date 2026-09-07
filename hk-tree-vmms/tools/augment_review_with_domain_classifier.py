#!/usr/bin/env python3
"""Add held-out-audited VMMS classifier suggestions to a generated review package."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exhaustive_species_drafts import canonical  # noqa: E402
from build_species_review_package import UNKNOWN, draw_overlay, write_csv  # noqa: E402


def crop(image: Image.Image, label: dict[str, Any]) -> Image.Image:
    width, height = image.size
    box = label.get("bbox") or {}
    if not all(k in box for k in ("left", "top", "right", "bottom")):
        points = label["points"]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        box = {"left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys)}
    x1, x2, y1, y2 = box["left"] * width, box["right"] * width, box["top"] * height, box["bottom"] * height
    pad = max(x2 - x1, y2 - y1) * .10
    bounds = (max(0, int(x1-pad)), max(0, int(y1-pad)), min(width, int(x2+pad+1)), min(height, int(y2+pad+1)))
    return image.crop(bounds).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    cfg = parser.parse_args()
    review = cfg.review_dir.resolve()
    model = YOLO(str(cfg.model.resolve()))
    mapping = json.loads(cfg.dataset_summary.read_text(encoding="utf-8"))["class_mapping"]
    folder_to_species = {folder: species for species, folder in mapping.items()}
    jobs: list[tuple[Path, dict[str, Any], dict[str, Any], Image.Image]] = []
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((review / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        image_path = review / "images" / f"{path.stem}.jpg"
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        for label in record.get("labels") or []:
            if label.get("species_confidence_tier") == "human_verified":
                continue
            jobs.append((path, record, label, crop(image, label)))
        loaded.append((path, record))
    for start in range(0, len(jobs), 128):
        batch = jobs[start:start+128]
        results = model.predict([job[3] for job in batch], imgsz=224, batch=128, device=0, verbose=False)
        for (_, _, label, _), result in zip(batch, results):
            order = result.probs.top5
            options = [{"species": folder_to_species[result.names[int(i)]], "confidence": float(result.probs.data[int(i)])} for i in order]
            hint = {"model": str(cfg.model.resolve()), "top1_species": options[0]["species"], "confidence": options[0]["confidence"], "top5": options,
                    "validation": "route-block-held-out top1=0.645; confidence>=0.99 selective accuracy=0.845"}
            inventory = label.get("inventory_match") or {}
            hint["agrees_with_inventory"] = bool(inventory.get("species") and canonical(inventory["species"]) == canonical(options[0]["species"]))
            label["vmms_domain_classifier_suggestion"] = hint
            if label.get("species") == UNKNOWN and not label.get("candidate_species"):
                label["candidate_species"] = options[0]["species"]
    # Multiple labels share each in-memory record object, so write every record once.
    seen: set[Path] = set()
    for path, record, _, _ in jobs:
        if path in seen:
            continue
        seen.add(path)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        overlay = review / "overlays" / record["dataset_route"] / f"{path.stem}.jpg"
        draw_overlay(review / "images" / f"{path.stem}.jpg", record.get("labels") or [], overlay, record["status"])

    queue_path = review / "instance_review_queue.csv"
    with queue_path.open(encoding="utf-8-sig", newline="") as stream:
        queue = list(csv.DictReader(stream))
    hints = {label.get("label_id"): label.get("vmms_domain_classifier_suggestion") for _, record in loaded for label in record.get("labels") or []}
    # loaded objects were mutated through jobs; enrich queue without changing formal classes.
    for row in queue:
        hint = hints.get(row["label_id"]) or {}
        row["domain_classifier_top1"] = hint.get("top1_species", "")
        row["domain_classifier_confidence"] = hint.get("confidence", "")
        row["domain_inventory_agreement"] = hint.get("agrees_with_inventory", "")
    write_csv(queue_path, queue)
    summary_path = review / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["vmms_domain_classifier"] = {
        "model": str(cfg.model.resolve()), "suggested_instances": len(jobs),
        "formal_labels_changed": 0, "held_out_top1_accuracy": 0.645320197044335,
        "held_out_accuracy_at_confidence_0.99": 0.845360824742268,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["vmms_domain_classifier"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
