#!/usr/bin/env python3
"""Train YOLO11 instance segmentation while keeping the test split unavailable."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "_cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "models" / "_cache" / "matplotlib"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, run_dir, model_path = args.data.resolve(), args.run_dir.resolve(), args.model.resolve()
    if not data.is_file() or not model_path.is_file():
        raise FileNotFoundError(data if not data.is_file() else model_path)
    if (run_dir / "training_complete.json").is_file():
        print(json.dumps({"status": "already_complete", "run_dir": str(run_dir)}))
        return
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = YOLO(str(model_path))
    model.train(
        data=str(data),
        task="segment",
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.workers,
        device=0,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        cos_lr=True,
        patience=10,
        close_mosaic=10,
        amp=True,
        deterministic=True,
        seed=args.seed,
        save=True,
        plots=True,
        verbose=True,
    )
    payload = {
        "status": "training_complete",
        "elapsed_seconds": time.time() - started,
        "model": str(model_path),
        "data": str(data),
        "best_checkpoint": str(run_dir / "weights" / "best.pt"),
        "last_checkpoint": str(run_dir / "weights" / "last.pt"),
        "test_loaded_during_training": False,
    }
    (run_dir / "training_complete.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
