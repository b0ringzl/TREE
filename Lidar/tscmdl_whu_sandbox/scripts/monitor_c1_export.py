"""Continuously display C1 full-export progress from its atomic progress file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    value = max(0, int(seconds))
    return f"{value // 3600:02d}:{(value % 3600) // 60:02d}:{value % 60:02d}"


def render(progress: dict[str, object], run_dir: Path) -> str:
    fraction = float(progress["progress_fraction"])
    width = 36
    filled = min(width, int(round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    return "\n".join(
        [
            "WHU-STree C1 Shared Asset Export",
            f"Run      : {run_dir}",
            f"State    : {progress['status']}",
            "",
            f"Progress : [{bar}] {100 * fraction:6.2f}%",
            (
                f"Tracks   : {progress['completed_trajectories']}/"
                f"{progress['total_trajectories']}"
            ),
            (
                f"Samples  : {progress['completed_samples']}/"
                f"{progress['total_samples']}"
            ),
            f"Current  : {progress['current_trajectory'] or '-'}",
            f"Elapsed  : {duration(progress['elapsed_seconds_this_run'])}",
            f"ETA      : {duration(progress['eta_seconds_this_run'])}",
            f"Output   : {int(progress['output_bytes']) / 1024**3:.2f} GiB",
            f"Free D   : {int(progress['free_disk_bytes']) / 1024**3:.2f} GiB",
            f"Updated  : {progress['updated_at']}",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    progress_path = args.run_dir / "progress.json"
    while True:
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            print("\x1b[2J\x1b[H" + render(progress, args.run_dir), flush=True)
            if args.once or progress["status"] in {"complete", "failed"}:
                break
        elif args.once:
            raise FileNotFoundError(progress_path)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
