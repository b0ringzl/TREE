"""Show YOLO training progress together with the current NVIDIA GPU state."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def gpu_state() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception as error:
        return f"unavailable: {error}"


def snapshot(run_dir: Path) -> dict[str, object]:
    results_path = run_dir / "results.csv"
    result: dict[str, object] = {"run_dir": str(run_dir.resolve()), "gpu": gpu_state()}
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            result["latest"] = {key.strip(): value.strip() for key, value in rows[-1].items()}
            result["epochs_recorded"] = len(rows)
    if (run_dir / "final_test_metrics.json").is_file():
        result["final"] = json.loads(
            (run_dir / "final_test_metrics.json").read_text(encoding="utf-8")
        )
    elif (run_dir / "training_complete.json").is_file():
        result["training"] = json.loads(
            (run_dir / "training_complete.json").read_text(encoding="utf-8")
        )
    else:
        result["status"] = "training_or_waiting"
    return result


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    while True:
        print(json.dumps(snapshot(run_dir), ensure_ascii=False, indent=2), flush=True)
        if not args.watch:
            break
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
