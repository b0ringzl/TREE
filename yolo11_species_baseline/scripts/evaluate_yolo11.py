"""Evaluate the selected YOLO11 checkpoint on test and export example predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: object) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def main() -> None:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.resolve()
    run_dir = args.run_dir.resolve()
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)

    with dataset_yaml.open("r", encoding="utf-8") as stream:
        dataset = yaml.safe_load(stream)
    dataset_root = Path(dataset["path"])
    test_images = dataset_root / dataset["test"]
    image_paths = sorted(path for path in test_images.iterdir() if path.is_file())
    if not image_paths:
        raise ValueError("Test image directory is empty")

    model = YOLO(str(best))
    started = time.time()
    metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=args.image_size,
        batch=32,
        device=0,
        workers=8,
        project=str(run_dir),
        name="test_validation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    box = metrics.box
    payload = {
        "status": "complete",
        "checkpoint": str(best),
        "checkpoint_sha256": sha256_file(best),
        "test_images": len(image_paths),
        "precision": scalar(box.mp),
        "recall": scalar(box.mr),
        "map50": scalar(box.map50),
        "map50_95": scalar(box.map),
        "per_class_map50_95": [float(value) for value in box.maps.tolist()],
        "class_names": dataset["names"],
        "elapsed_seconds": time.time() - started,
        "cuda_peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "test_started_after_training_complete": (run_dir / "training_complete.json").is_file(),
    }

    prediction_dir = run_dir / "test_predictions"
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.image_size,
        conf=args.confidence,
        device=0,
        batch=32,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(run_dir),
        name=prediction_dir.name,
        exist_ok=True,
        verbose=False,
    )
    prediction_rows: list[dict[str, object]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for xyxy, confidence, class_index in zip(
            boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
        ):
            prediction_rows.append(
                {
                    "image": Path(result.path).name,
                    "class_index": int(class_index),
                    "species": result.names[int(class_index)],
                    "confidence": float(confidence),
                    "x0": float(xyxy[0]),
                    "y0": float(xyxy[1]),
                    "x1": float(xyxy[2]),
                    "y1": float(xyxy[3]),
                }
            )
    with (run_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["image", "class_index", "species", "confidence", "x0", "y0", "x1", "y1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    payload["prediction_count_at_confidence"] = len(prediction_rows)
    payload["predicted_image_count_at_confidence"] = len(
        {str(row["image"]) for row in prediction_rows}
    )
    class_prediction_counts = Counter(str(row["species"]) for row in prediction_rows)
    payload["prediction_counts_by_class"] = {
        str(name): class_prediction_counts.get(str(name), 0)
        for name in dataset["names"].values()
    }
    payload["prediction_confidence"] = args.confidence
    (run_dir / "final_test_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
