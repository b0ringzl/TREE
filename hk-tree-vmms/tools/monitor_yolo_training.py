#!/usr/bin/env python
"""Display live GPU and metric progress for the current merged-tree YOLO run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "derived" / "training_runs" / "combined_tree_segmentation"
ACTIVE_RUN_FILE = RUNS_ROOT / "active_run.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def process_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except Exception:
        return False


def gpu_status() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, encoding="utf-8", errors="replace").strip()
        name, util, used, total, temp, power = [item.strip() for item in output.splitlines()[0].split(",")]
        percent = 100.0 * float(used) / max(float(total), 1.0)
        return (
            f"{name} | GPU {util}% | VRAM {used}/{total} MiB ({percent:.1f}%) | "
            f"{temp} C | {power} W"
        )
    except Exception as exc:
        return f"GPU status unavailable: {exc}"


def latest_metrics(results_csv: Path) -> tuple[int, dict[str, str]] | None:
    if not results_csv.is_file():
        return None
    try:
        with results_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return None
        row = {key.strip(): value.strip() for key, value in rows[-1].items() if key is not None}
        return len(rows), row
    except Exception:
        return None


def metric(row: dict[str, str], *needles: str) -> str:
    for key, value in row.items():
        normal = key.lower().replace(" ", "")
        if all(needle.lower() in normal for needle in needles):
            return value
    return "-"


def render() -> tuple[str, bool]:
    marker = read_json(ACTIVE_RUN_FILE)
    if not marker:
        return f"No run marker found at {ACTIVE_RUN_FILE}", False

    pid = marker.get("pid")
    marker_status = str(marker.get("status", "unknown"))
    running = marker_status == "running" and process_is_running(pid)
    run_dir = Path(marker.get("run_dir", ""))
    results_csv = run_dir / "results.csv"
    result = latest_metrics(results_csv)

    lines = [
        "Three-route tree segmentation training monitor",
        f"Updated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Status  : {marker_status} | PID {pid} | process {'running' if running else 'not running'}",
        f"Run     : {run_dir}",
        f"Config  : {marker.get('model')} | imgsz={marker.get('imgsz')} | batch={marker.get('batch')} | epochs={marker.get('epochs')}",
        f"GPU     : {gpu_status()}",
    ]

    if result:
        epoch_count, row = result
        lines.extend(
            [
                "",
                f"Progress: epoch {epoch_count}/{marker.get('epochs', '?')}",
                f"Segment : mAP50={metric(row, 'metrics/map50(m)')} | mAP50-95={metric(row, 'metrics/map50-95(m)')}",
                f"Box     : mAP50={metric(row, 'metrics/map50(b)')} | mAP50-95={metric(row, 'metrics/map50-95(b)')}",
                f"Loss    : train box={metric(row, 'train/box_loss')} seg={metric(row, 'train/seg_loss')} cls={metric(row, 'train/cls_loss')}",
                f"          val box={metric(row, 'val/box_loss')} seg={metric(row, 'val/seg_loss')} cls={metric(row, 'val/cls_loss')}",
            ]
        )
    else:
        lines.extend(["", "Progress: waiting for the first completed epoch..."])

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    lines.append(
        f"Weights : best={'yes' if best.is_file() else 'no'} | last={'yes' if last.is_file() else 'no'}"
    )
    if not running and marker_status == "running":
        lines.append("Warning : marker says running but the process is no longer alive; inspect the training console/log.")
    return "\n".join(lines), running


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    while True:
        output, running = render()
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(output, flush=True)
        if not args.watch or not running:
            return 0
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
