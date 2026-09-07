#!/usr/bin/env python
"""Train a YOLO11 segmentation baseline on the merged three-route dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "derived" / "combined_tree_segmentation"
RUNS_ROOT = PROJECT_ROOT / "derived" / "training_runs" / "combined_tree_segmentation"
ACTIVE_RUN_FILE = RUNS_ROOT / "active_run.json"


def latest_dataset() -> Path:
    candidates = sorted(
        (
            path
            for path in DATASET_ROOT.glob("effective_yolo_*")
            if (path / "data.yaml").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No merged segmentation dataset found under {DATASET_ROOT}")
    return candidates[0]


def process_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_marker(payload: dict[str, Any]) -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = ACTIVE_RUN_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(ACTIVE_RUN_FILE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to data.yaml; latest merged export is used by default")
    parser.add_argument("--model", default="yolo11m-seg.pt", help="Ultralytics model name or local weight path")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--batch",
        type=float,
        default=5.0,
        help="Fixed batch size; a value between 0 and 1 enables Ultralytics AutoBatch",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", help="Explicit run name; a timestamped name is used by default")
    parser.add_argument("--dry-run", action="store_true", help="Validate the setup without starting training")
    parser.add_argument("--allow-concurrent", action="store_true", help="Allow another marked run to coexist")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    batch_value: int | float = (
        int(args.batch) if args.batch >= 1.0 and args.batch.is_integer() else args.batch
    )
    data_yaml = args.data.resolve() if args.data else latest_dataset() / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)

    previous = read_json(ACTIVE_RUN_FILE)
    if (
        not args.allow_concurrent
        and previous.get("status") == "running"
        and process_is_running(previous.get("pid"))
    ):
        print(
            f"A training run is already active (PID {previous['pid']}): {previous.get('run_dir', '')}",
            file=sys.stderr,
        )
        return 2

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Unable to import the CUDA training stack: {exc}", file=sys.stderr)
        return 3

    if not torch.cuda.is_available():
        print("CUDA is not available; refusing to start an unintended CPU training run.", file=sys.stderr)
        return 4

    run_name = args.name or f"yolo11m_seg_1024_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = RUNS_ROOT / run_name
    configuration: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "running",
        "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data": str(data_yaml),
        "dataset_dir": str(data_yaml.parent),
        "model": args.model,
        "run_dir": str(run_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch_value,
        "workers": args.workers,
        "patience": args.patience,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    print(json.dumps(configuration, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    write_marker(configuration)
    os.chdir(WORKSPACE_ROOT)

    try:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            project=str(RUNS_ROOT),
            name=run_name,
            exist_ok=True,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch_value,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            pretrained=True,
            amp=True,
            cache=False,
            seed=42,
            deterministic=True,
            cos_lr=True,
            close_mosaic=10,
            plots=True,
            save=True,
            save_period=5,
            verbose=True,
        )
        configuration.update(
            {
                "status": "completed",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "best_weight": str(run_dir / "weights" / "best.pt"),
                "last_weight": str(run_dir / "weights" / "last.pt"),
                "results_csv": str(run_dir / "results.csv"),
            }
        )
        write_marker(configuration)
        print(f"Training completed: {run_dir}")
        return 0
    except KeyboardInterrupt:
        configuration.update(
            {
                "status": "interrupted",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        write_marker(configuration)
        print("Training interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        configuration.update(
            {
                "status": "failed",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_marker(configuration)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
