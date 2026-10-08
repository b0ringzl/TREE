"""Read-only monitor for the NJ YOLO11 classification comparison."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_dir = parse_args().run_dir.resolve()
    results_path = run_dir / "results.csv"
    latest = None
    row_count = 0
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        row_count = len(rows)
        latest = rows[-1] if rows else None
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception as error:
        gpu = f"unavailable: {error}"
    payload = {
        "run_dir": str(run_dir),
        "gpu": gpu,
        "epochs_recorded": row_count,
        "latest": latest,
        "training_complete": (run_dir / "training_complete.json").is_file(),
        "selection_complete": (run_dir / "selection_complete.json").is_file(),
        "comparison_complete": (run_dir / "comparison.json").is_file(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
