#!/usr/bin/env python3
"""Evaluate a validation-selected YOLO11 segmentation checkpoint on held-out test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_metrics(metric: object) -> dict[str, object]:
    return {
        "precision": float(metric.mp),
        "recall": float(metric.mr),
        "map50": float(metric.map50),
        "map75": float(metric.map75),
        "map50_95": float(metric.map),
        "per_class_map50_95": [float(value) for value in metric.maps],
    }


def main() -> None:
    args = parse_args()
    data, summary_path, run_dir = args.data.resolve(), args.dataset_summary.resolve(), args.run_dir.resolve()
    output_path = run_dir / "final_test_metrics.json"
    if output_path.is_file():
        print(output_path.read_text(encoding="utf-8"))
        return
    training_complete = run_dir / "training_complete.json"
    checkpoint = run_dir / "weights" / "best.pt"
    if not training_complete.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(training_complete if not training_complete.is_file() else checkpoint)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(data),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=0,
        plots=True,
        project=str(run_dir),
        name="held_out_test_evaluation",
        exist_ok=True,
        verbose=True,
    )
    payload = {
        "status": "complete",
        "model": "YOLO11m-seg",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "test_images": summary["split_images"]["test"],
        "test_instances": summary["split_instances"]["test"],
        "class_names": summary["class_names"],
        "box": task_metrics(metrics.box),
        "mask": task_metrics(metrics.seg),
        "test_evaluated_after_validation_selection": True,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    concise = {
        "status": "complete",
        "test_images": payload["test_images"],
        "test_instances": payload["test_instances"],
        "box": {key: payload["box"][key] for key in ("precision", "recall", "map50", "map50_95")},
        "mask": {key: payload["mask"][key] for key in ("precision", "recall", "map50", "map50_95")},
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
