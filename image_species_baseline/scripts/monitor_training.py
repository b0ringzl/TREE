#!/usr/bin/env python3
"""Read-only monitor for the independent ten-class ResNet50 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def gpu_status() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nvidia-smi unavailable"


def snapshot(run_dir: Path) -> None:
    print(f"run_dir: {run_dir}")
    print(f"gpu: {gpu_status()}")
    final = read_json(run_dir / "final_metrics.json")
    if final:
        tree = final["metrics"]["test_tree_group"]
        crop = final["metrics"]["test_crop"]
        print(
            "status=complete "
            f"best_epoch={final['best_epoch']} "
            f"test_tree_acc={tree['accuracy']:.4f} "
            f"test_tree_macro_f1={tree['macro_f1']:.4f} "
            f"test_crop_acc={crop['accuracy']:.4f} "
            f"test_crop_macro_f1={crop['macro_f1']:.4f}"
        )
        return
    live = read_json(run_dir / "live_progress.json")
    if live:
        print(json.dumps(live, ensure_ascii=False, indent=2))
    history_path = run_dir / "training_history.csv"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            row = rows[-1]
            print(
                f"last_epoch={row['epoch']} train_loss={float(row['train_loss']):.4f} "
                f"val_group_f1={float(row['val_group_macro_f1']):.4f} "
                f"peak_vram={float(row['peak_allocated_mib']):.0f}MiB"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    while True:
        snapshot(args.run_dir.resolve())
        if not args.watch:
            break
        print("-" * 72, flush=True)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
