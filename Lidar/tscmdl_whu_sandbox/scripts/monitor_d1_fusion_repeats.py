"""Read-only monitor for the D1 frozen-backbone fusion suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "temporarily_unreadable"}


def gpu_line() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def render(suite: Path) -> str:
    state = read_json(suite / "suite_state.json")
    protocol = read_json(suite / "protocol.json")
    seeds = [int(value) for value in protocol.get("seeds", [])]
    lines = [
        f"time={datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"suite_status={state.get('status', 'not_started')} phase={state.get('phase', '-')}",
        f"active_seed={state.get('active_seed', 0)} completed={state.get('completed_seeds', [])}",
        f"message={state.get('message', '-')}",
        f"gpu={gpu_line()}",
    ]
    for seed in seeds:
        feature = read_json(suite / "features" / f"seed_{seed}" / "manifest.json")
        run_dir = suite / "runs" / f"seed_{seed}"
        run_state = read_json(run_dir / "run_state.json")
        progress = read_json(run_dir / "live_progress.json")
        metrics = read_json(run_dir / "final_metrics.json")
        if metrics.get("status") == "complete":
            val = metrics["metrics"]["val"]
            test = metrics["metrics"]["test"]
            detail = (
                f"complete best={metrics.get('best_epoch')} "
                f"valF1={100*float(val['macro_f1']):.2f}% "
                f"testF1={100*float(test['macro_f1']):.2f}%"
            )
        elif progress:
            detail = (
                f"{run_state.get('status', 'running')} "
                f"epoch={progress.get('epoch', 0)}/{progress.get('total_epochs', 0)} "
                f"best={progress.get('best_epoch', 0)} "
                f"valF1={100*float(progress.get('val_macro_f1', 0.0)):.2f}%"
            )
        else:
            detail = str(run_state.get("status", "waiting"))
        lines.append(
            f"seed={seed} cache={feature.get('status', 'waiting')} run={detail}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    suite = args.suite_root.resolve()
    while True:
        print(render(suite), flush=True)
        if not args.watch:
            return
        state = read_json(suite / "suite_state.json")
        if state.get("status") in {"complete", "failed", "paused"}:
            return
        print("-" * 72, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
