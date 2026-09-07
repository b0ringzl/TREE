"""Read-only aggregate monitor for the D1 four-class PTv2 repeat suite."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from monitor_training import build_snapshot, format_duration, gpu_snapshot, progress_bar  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--exit-on-complete", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            if attempt == 2:
                return None
            time.sleep(0.05)
    return None


def percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def build_suite_snapshot(suite_root: Path) -> dict[str, Any]:
    protocol = read_json(suite_root / "protocol.json") or {}
    state = read_json(suite_root / "suite_state.json") or {}
    seeds = [int(seed) for seed in protocol.get("seeds", [])]
    active_seed = int(state.get("active_seed", 0) or 0)
    per_seed: dict[str, Any] = {}
    completed = 0
    progress = 0.0
    for seed in seeds:
        snapshot = build_snapshot(suite_root / "runs" / f"seed_{seed}")
        per_seed[str(seed)] = snapshot
        final = snapshot.get("final")
        if final and final.get("status") == "complete":
            completed += 1
            progress += 1.0
        elif seed == active_seed:
            progress += float(snapshot.get("progress_fraction", 0.0))
    fraction = progress / len(seeds) if seeds else 0.0
    complete = bool(seeds) and completed == len(seeds)
    return {
        "suite_root": str(suite_root),
        "status": "complete" if complete else str(state.get("status", "waiting")),
        "active_seed": active_seed,
        "seeds": seeds,
        "completed_seed_count": completed,
        "progress_fraction": fraction,
        "protocol": protocol,
        "suite_state": state,
        "per_seed": per_seed,
        "gpu": gpu_snapshot(),
        "complete": complete,
    }


def render(snapshot: dict[str, Any]) -> str:
    protocol = snapshot["protocol"]
    state = snapshot["suite_state"]
    seeds = snapshot["seeds"]
    lines = [
        "D1 Four-Class PTv2 Three-Seed Monitor",
        f"Root     : {snapshot['suite_root']}",
        f"State    : {snapshot['status']}",
        (
            "Setup    : "
            f"batch={protocol.get('hyperparameters', {}).get('batch_size')}  "
            f"eval_batch={protocol.get('hyperparameters', {}).get('eval_batch_size')}  "
            f"epochs={protocol.get('hyperparameters', {}).get('epochs')}  "
            f"balanced={protocol.get('hyperparameters', {}).get('balanced_sampler')}"
        ),
        f"Seeds    : {', '.join(str(seed) for seed in seeds)}",
        "",
        (
            f"Overall  : {progress_bar(float(snapshot['progress_fraction']))} "
            f"{snapshot['completed_seed_count']}/{len(seeds)} complete  "
            f"({100.0 * float(snapshot['progress_fraction']):.2f}%)"
        ),
    ]
    if snapshot["active_seed"]:
        lines.append(f"Active   : seed {snapshot['active_seed']}")
    if state.get("message"):
        lines.append(f"Message  : {state['message']}")
    lines.extend(["", "Per-seed status"])
    for seed in seeds:
        item = snapshot["per_seed"][str(seed)]
        final = item.get("final")
        run_state = item.get("run_state") or {}
        if final and final.get("status") == "complete":
            lines.append(
                f"  {seed}: complete  best={int(final['best_epoch'])}  "
                f"val_f1={percent(final['metrics']['val']['macro_f1'])}  "
                f"test_f1={percent(final['metrics']['test']['macro_f1'])}"
            )
        elif run_state.get("status") == "paused":
            lines.append(
                f"  {seed}: paused after epoch={int(run_state.get('saved_epoch', 0))}"
            )
        elif item.get("configured_args"):
            lines.append(
                f"  {seed}: {item['state']}  epoch={item['epoch']}/"
                f"{item['total_epochs']}  progress={item['progress_percent']:.2f}%"
            )
        else:
            lines.append(f"  {seed}: pending")

    active = snapshot["per_seed"].get(str(snapshot["active_seed"]))
    if active and active.get("configured_args"):
        latest = active.get("latest")
        live = active.get("live") or {}
        lines.extend(
            [
                "",
                (
                    f"Current  : epoch {active['epoch']}/{active['total_epochs']}  "
                    f"elapsed={format_duration(active['elapsed_seconds'])}  "
                    f"ETA={format_duration(active['eta_seconds'])}"
                ),
            ]
        )
        if live.get("status") in {"running", "phase_complete"}:
            lines.append(
                f"Phase    : {live.get('phase', '-')}  "
                f"batch={live.get('batch', 0)}/{live.get('total_batches', 0)}  "
                f"ETA={format_duration(live.get('phase_eta_seconds'))}"
            )
        if latest:
            lines.append(
                f"Latest   : train_acc={percent(latest['train_accuracy'])}  "
                f"train_f1={percent(latest['train_macro_f1'])}"
            )
            lines.append(
                f"Val      : acc={percent(latest['val_accuracy'])}  "
                f"macro_f1={percent(latest['val_macro_f1'])}"
            )
        best = active.get("best")
        if best:
            lines.append(
                f"Best val : epoch {int(best['epoch'])}  "
                f"acc={percent(best['val_accuracy'])}  "
                f"macro_f1={percent(best['val_macro_f1'])}"
            )
    gpu = snapshot.get("gpu")
    if gpu:
        lines.extend(
            [
                "",
                (
                    f"GPU      : {gpu['name']}  util={gpu['utilization_percent']}%  "
                    f"memory={gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB  "
                    f"temp={gpu['temperature_c']} C"
                ),
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    while True:
        snapshot = build_suite_snapshot(suite_root)
        if args.json:
            print(json.dumps(snapshot, ensure_ascii=False, indent=2), flush=True)
        else:
            if not args.once:
                os.system("cls" if os.name == "nt" else "clear")
            print(render(snapshot), flush=True)
        if args.once or (args.exit_on_complete and snapshot["complete"]):
            break
        print("\nPress Ctrl+C to close monitoring; training continues.", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
