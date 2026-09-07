"""Train a reproducible YOLO11 detection baseline on the ten-class dataset."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception as error:
        return f"unavailable: {error}"


def main() -> None:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.resolve()
    run_dir = args.run_dir.resolve()
    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    if (run_dir / "weights" / "best.pt").exists():
        raise FileExistsError(f"Training output already exists: {run_dir / 'weights' / 'best.pt'}")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "nvidia_smi": gpu_snapshot(),
        "model": args.model,
        "dataset_yaml": str(dataset_yaml),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    started = time.time()
    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=0,
        workers=8,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.learning_rate,
        seed=args.seed,
        deterministic=True,
        patience=20,
        cos_lr=True,
        close_mosaic=10,
        mosaic=0.5,
        translate=0.05,
        scale=0.3,
        hsv_s=0.3,
        hsv_v=0.2,
        amp=True,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )
    completed = {
        "status": "training_complete",
        "elapsed_seconds": time.time() - started,
        "best_checkpoint": str(run_dir / "weights" / "best.pt"),
        "last_checkpoint": str(run_dir / "weights" / "last.pt"),
    }
    (run_dir / "training_complete.json").write_text(
        json.dumps(completed, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
