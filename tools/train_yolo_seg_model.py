#!/usr/bin/env python3
"""Train a YOLO instance-segmentation model with project-local cache/output paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE = PROJECT_ROOT / "models" / "_cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("TMP", str(PROJECT_CACHE / "tmp"))
os.environ.setdefault("TEMP", str(PROJECT_CACHE / "tmp"))
os.environ.setdefault("WINDIR", os.environ.get("SystemRoot", r"C:\Windows"))
os.environ.setdefault("SystemRoot", os.environ.get("WINDIR", r"C:\Windows"))
# Disable Ultralytics' Unicode progress bars; use tools/watch_yolo_training.ps1 for clean live progress.
os.environ.setdefault("YOLO_VERBOSE", "false")
for cache_dir in {
    Path(os.environ["TORCH_HOME"]),
    Path(os.environ["HF_HOME"]),
    Path(os.environ["XDG_CACHE_HOME"]),
    Path(os.environ["MPLCONFIGDIR"]),
    Path(os.environ["TMP"]),
}:
    cache_dir.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer


def read_results_csv_without_polars(self) -> dict:
    import csv

    csv_path = Path(self.csv)
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return rows[-1]


BaseTrainer.read_results_csv = read_results_csv_without_polars
BaseTrainer.plot_metrics = lambda self: None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO model for single-tree mask proposals.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--project", type=Path, default=Path("models/yolo_runs"))
    parser.add_argument("--name", default="urban_tree_seg_v1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Resume from a YOLO checkpoint.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint path for --resume.")
    args = parser.parse_args()

    if args.resume:
        checkpoint = args.resume_from or args.project / args.name / "weights" / "last.pt"
        model = YOLO(str(checkpoint.resolve()))
        model.train(resume=True)
        return 0

    if args.data is None:
        parser.error("--data is required unless --resume is used.")

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        cache=False,
        task="segment",
        amp=args.amp,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
