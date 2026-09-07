"""Read-only monitor for the queued D1 clean-image ResNet50 seed suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "scripts"))

import monitor_training  # noqa: E402


DEFAULT_SUITE = (
    PROJECT_ROOT
    / "lidar data"
    / "whu"
    / "derived"
    / "tscmdl"
    / "d1_resnet50_clean_repeats"
    / "20260817_protocol_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def suite_snapshot(suite_root: Path) -> dict[str, Any]:
    protocol = read_json(suite_root / "protocol.json")
    state = read_json(suite_root / "suite_state.json")
    seeds = [int(seed) for seed in protocol.get("seeds", [20260728, 20260729, 20260730])]
    pending_seeds = {int(seed) for seed in state.get("pending_seeds", [])}
    active_seeds = {int(seed) for seed in state.get("current_seeds", [])}
    if state.get("current_seed") is not None:
        active_seeds.add(int(state["current_seed"]))
    runs = []
    for seed in seeds:
        run_dir = suite_root / "runs" / f"seed_{seed}"
        snapshot = monitor_training.build_snapshot(run_dir)
        # A failed state from an earlier launch attempt is historical once the
        # queue has explicitly scheduled the seed for a clean retry.
        if seed in pending_seeds and seed not in active_seeds:
            snapshot["state"] = "queued"
            snapshot["stale_seconds"] = None
        snapshot["seed"] = seed
        runs.append(snapshot)
    return {
        "suite_root": str(suite_root),
        "suite_status": state.get("status", "waiting_for_runner"),
        "current_seed": state.get("current_seed"),
        "current_seeds": state.get("current_seeds", []),
        "pending_seeds": state.get("pending_seeds", []),
        "completed_seeds": state.get("completed_seeds", []),
        "updated_at": state.get("updated_at"),
        "protocol": protocol,
        "runs": runs,
    }


def metric(row: dict[str, Any] | None, name: str) -> str:
    if not row or name not in row:
        return "-"
    return f"{100.0 * float(row[name]):.2f}%"


def render(snapshot: dict[str, Any]) -> str:
    lines = [
        "D1 Clean Image ResNet50 - Three Seed Monitor",
        f"Suite   : {snapshot['suite_root']}",
        f"Status  : {snapshot['suite_status']}",
        f"Current : {snapshot['current_seeds'] or snapshot['current_seed']}",
        f"Updated : {snapshot['updated_at']}",
        "",
        "seed      state             epoch       progress   val_MF1   best_MF1  age",
    ]
    for run in snapshot["runs"]:
        latest = run.get("latest")
        best = run.get("best")
        age = monitor_training.format_duration(run.get("stale_seconds"))
        lines.append(
            f"{run['seed']:<9} {run['state']:<17} "
            f"{run['epoch']:>3}/{run['total_epochs']:<3}   "
            f"{run['progress_percent']:>7.2f}%   "
            f"{metric(latest, 'val_macro_f1'):>8}   "
            f"{metric(best, 'val_macro_f1'):>8}  {age}"
        )
    active_seeds = set(snapshot.get("current_seeds") or [])
    if not active_seeds and snapshot.get("current_seed") is not None:
        active_seeds.add(snapshot["current_seed"])
    active_runs = [run for run in snapshot["runs"] if run["seed"] in active_seeds]
    for current in active_runs:
        live = current.get("live") or {}
        lines.extend(
            [
                "",
                f"Seed {current['seed']} live: {live.get('phase', '-')} "
                f"batch={live.get('batch', '-')}/{live.get('total_batches', '-')} "
                f"loss={live.get('loss', '-')} "
                f"acc={metric(live, 'accuracy')} "
                f"macro_f1={metric(live, 'macro_f1')}",
                f"Phase ETA : {monitor_training.format_duration(live.get('phase_eta_seconds'))}",
                f"Peak VRAM : {float(live.get('peak_allocated_mib', 0.0)):.0f} MiB",
            ]
        )
    gpu = active_runs[0].get("gpu_live") if active_runs else None
    if gpu:
        lines.append(
            f"GPU       : {gpu['name']} util={gpu['utilization_percent']}% "
            f"memory={gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB "
            f"temp={gpu['temperature_c']} C"
        )
    lines.extend(
        [
            "",
            "Selection: validation Macro-F1, then validation accuracy.",
            "Test split remains locked until each seed finishes training.",
            "Ctrl+C closes this read-only monitor and does not stop training.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    while True:
        snapshot = suite_snapshot(suite_root)
        if args.json:
            print(json.dumps(snapshot, ensure_ascii=False))
        else:
            print("\f" + render(snapshot), flush=True)
        if args.once or snapshot["suite_status"] in {"complete", "failed", "paused"}:
            return
        time.sleep(max(args.interval, 0.5))


if __name__ == "__main__":
    main()
