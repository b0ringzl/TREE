#!/usr/bin/env python3
"""Evaluate selective confidence accuracy for a YOLO classification model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cfg = parser.parse_args()
    paths = sorted(cfg.val_dir.resolve().glob("*/*.jpg"))
    model = YOLO(str(cfg.model.resolve()))
    rows = []
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    for result in model.predict(paths, imgsz=224, batch=128, device=0, verbose=False):
        truth = Path(result.path).parent.name
        top1 = result.names[int(result.probs.top1)]
        confidence = float(result.probs.top1conf)
        correct = truth == top1
        rows.append({"path": result.path, "truth": truth, "prediction": top1, "confidence": confidence, "correct": correct})
        per_class[truth]["support"] += 1
        per_class[truth]["correct"] += int(correct)
    thresholds = []
    for threshold in (0, .50, .60, .70, .80, .85, .90, .95, .98, .99):
        selected = [row for row in rows if row["confidence"] >= threshold]
        correct = sum(row["correct"] for row in selected)
        thresholds.append({
            "threshold": threshold, "selected": len(selected),
            "coverage": len(selected) / len(rows) if rows else 0,
            "accuracy": correct / len(selected) if selected else 0,
        })
    report = {
        "model": str(cfg.model.resolve()), "validation_images": len(rows),
        "top1_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "selective_accuracy": thresholds,
        "per_class": {name: {**count, "accuracy": count["correct"] / count["support"]} for name, count in per_class.items()},
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with cfg.output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
