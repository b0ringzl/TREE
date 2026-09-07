"""Real-time aggregate monitor for the portable C3 three-seed run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from monitor_training import build_snapshot, format_duration, gpu_snapshot, progress_bar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--exit-on-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def seed_line(seed: int, snapshot: dict[str, Any], *, paused: bool = False) -> str:
    final = snapshot.get("final")
    if final and final.get("status") == "complete":
        val = final["metrics"]["val"]
        test = final["metrics"]["test"]
        return (
            f"{seed}: complete  best={int(final['best_epoch'])}  "
            f"val_f1={percent(val['macro_f1'])}  "
            f"test_f1={percent(test['macro_f1'])}"
        )
    if snapshot["state"] == "failed":
        return f"{seed}: failed; rerun START_C3_TRAINING.bat after review"
    if paused:
        latest = snapshot.get("latest") or {}
        saved_epoch = int(latest.get("epoch", snapshot.get("epoch", 0)))
        return (
            f"{seed}: paused after epoch={saved_epoch}/"
            f"{snapshot.get('total_epochs', 0)}"
        )
    if snapshot.get("configured_args"):
        return (
            f"{seed}: {snapshot['state']}  "
            f"epoch={snapshot['epoch']}/{snapshot['total_epochs']}  "
            f"progress={snapshot['progress_percent']:.2f}%"
        )
    return f"{seed}: pending"


def render(root: Path) -> tuple[str, bool]:
    config = read_json(root / "c3_config.json") or {}
    seeds = [int(seed) for seed in config.get("seeds", [])]
    launcher = read_json(root / "output" / "c3_launcher_state.json") or {}
    snapshots = []
    progress = 0.0
    complete_count = 0
    active_snapshot = None
    active_seed = int(launcher.get("active_seed", 0) or 0)

    for seed in seeds:
        snapshot = build_snapshot(root / "output" / "runs" / f"seed_{seed}")
        snapshots.append((seed, snapshot))
        complete = bool(
            snapshot.get("final")
            and snapshot["final"].get("status") == "complete"
        )
        if complete:
            complete_count += 1
            progress += 1.0
        elif seed == active_seed:
            progress += float(snapshot.get("progress_fraction", 0.0))
            active_snapshot = snapshot

    fraction = progress / len(seeds) if seeds else 0.0
    all_complete = bool(seeds) and complete_count == len(seeds)
    state = "complete" if all_complete else str(launcher.get("status", "waiting"))
    lines = [
        "WHU-STree C3 Three-Seed Monitor",
        f"Root     : {root}",
        f"State    : {state}",
        (
            "Setup    : "
            f"batch={config.get('batch_size')}  "
            f"eval_batch={config.get('eval_batch_size')}  "
            f"epochs={config.get('epochs')}  "
            f"patience={config.get('patience')}  AMP=on"
        ),
        f"Seeds    : {', '.join(str(seed) for seed in seeds)}",
        "",
        (
            f"Overall  : {progress_bar(fraction)} "
            f"{complete_count}/{len(seeds)} complete  ({100*fraction:.2f}%)"
        ),
    ]

    if active_seed:
        lines.append(
            f"Active   : seed {active_seed} "
            f"({int(launcher.get('seed_index', 0) or 0)}/{len(seeds)})"
        )
    if launcher.get("message"):
        lines.append(f"Message  : {launcher['message']}")

    lines.extend(["", "Per-seed status"])
    lines.extend(
        f"  {seed_line(seed, snapshot, paused=state == 'paused' and seed == active_seed)}"
        for seed, snapshot in snapshots
    )

    if active_snapshot and active_snapshot.get("configured_args"):
        latest = active_snapshot.get("latest")
        live = active_snapshot.get("live") or {}
        displayed_epoch = (
            int(latest["epoch"])
            if state == "paused" and latest
            else active_snapshot["epoch"]
        )
        displayed_eta = (
            "paused"
            if state == "paused"
            else format_duration(active_snapshot["eta_seconds"])
        )
        lines.extend(
            [
                "",
                (
                    f"Current  : epoch {displayed_epoch}/"
                    f"{active_snapshot['total_epochs']}  "
                    f"elapsed={format_duration(active_snapshot['elapsed_seconds'])}  "
                    f"ETA={displayed_eta}"
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
            lines.extend(
                [
                    (
                        f"Latest   : train_acc={percent(latest['train_accuracy'])}  "
                        f"train_f1={percent(latest['train_macro_f1'])}"
                    ),
                    (
                        f"Val      : acc={percent(latest['val_accuracy'])}  "
                        f"macro_f1={percent(latest['val_macro_f1'])}"
                    ),
                ]
            )
        best = active_snapshot.get("best")
        if best:
            lines.append(
                f"Best val : epoch {int(best['epoch'])}  "
                f"acc={percent(best['val_accuracy'])}  "
                f"macro_f1={percent(best['val_macro_f1'])}"
            )

    gpu = gpu_snapshot()
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
    if state == "failed":
        lines.extend(
            [
                "",
                "The runner stopped after a failure; completed runs are preserved.",
                "Review output logs, then rerun START_C3_TRAINING.bat to resume.",
            ]
        )
    return "\n".join(lines), all_complete


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    while True:
        text, complete = render(root)
        os.system("cls" if os.name == "nt" else "clear")
        print(text, flush=True)
        if args.once or (args.exit_on_complete and complete):
            break
        print("\nPress Ctrl+C to close monitoring; training continues.", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
