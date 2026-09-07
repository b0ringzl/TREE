"""Read-only real-time monitor for WHU-STree classification training runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--exit-on-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", action="store_true")
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


def gpu_snapshot() -> dict[str, str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_line = result.stdout.strip().splitlines()[0]
    values = [value.strip() for value in first_line.split(",")]
    if len(values) != 5:
        return None
    return dict(
        zip(
            ("name", "utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c"),
            values,
        )
    )


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(fraction: float, width: int = 36) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(width * fraction))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def newest_artifact_time(run_dir: Path) -> float | None:
    candidates = [
        run_dir / "training_history.json",
        run_dir / "live_progress.json",
        run_dir / "run_state.json",
        run_dir / "launcher_state.json",
        run_dir / "last.pt",
        run_dir / "train.log",
        run_dir / "final_metrics.json",
        run_dir / "smoke_result.json",
    ]
    timestamps = [path.stat().st_mtime for path in candidates if path.is_file()]
    return max(timestamps) if timestamps else None


def build_snapshot(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "run_config.json") or {}
    history = read_json(run_dir / "training_history.json") or []
    live = read_json(run_dir / "live_progress.json") or {}
    run_state = read_json(run_dir / "run_state.json") or {}
    launcher_state = read_json(run_dir / "launcher_state.json") or {}
    final = read_json(run_dir / "final_metrics.json")
    smoke = read_json(run_dir / "smoke_result.json")
    configured_epochs = int(config.get("args", {}).get("epochs", 0) or 0)
    completed_epochs = len(history)
    total_epochs = max(configured_epochs, completed_epochs)
    latest = history[-1] if history else None
    recent = history[-min(10, len(history)) :] if history else []
    recent_epoch_seconds = (
        sum(float(item["epoch_seconds"]) for item in recent) / len(recent)
        if recent
        else None
    )
    remaining_epochs = max(total_epochs - completed_epochs, 0)
    eta_seconds = (
        recent_epoch_seconds * remaining_epochs
        if recent_epoch_seconds is not None
        else None
    )
    elapsed_seconds = (
        sum(float(item["epoch_seconds"]) for item in history) if history else 0.0
    )
    best = (
        max(
            history,
            key=lambda item: (
                float(item["val_macro_f1"]),
                float(item["val_accuracy"]),
            ),
        )
        if history
        else None
    )
    patience = int(config.get("args", {}).get("patience", 0) or 0)
    best_epoch = int(best["epoch"]) if best else 0
    epochs_without_improvement = (
        max(completed_epochs - best_epoch, 0) if best_epoch else 0
    )
    newest_time = newest_artifact_time(run_dir)
    stale_seconds = time.time() - newest_time if newest_time is not None else None
    complete = bool(final and final.get("status") == "complete")
    smoke_complete = bool(
        smoke
        and smoke.get("status") == "passed"
        and run_state.get("status") == "smoke_complete"
    )
    if complete:
        total_epochs = max(
            completed_epochs,
            int(final.get("epochs_completed", completed_epochs) or completed_epochs),
        )
    stale_threshold = max(
        90.0,
        3.0 * recent_epoch_seconds if recent_epoch_seconds is not None else 90.0,
    )
    live_epoch = int(live.get("epoch", 0) or 0)
    live_batch = int(live.get("batch", 0) or 0)
    live_total_batches = int(live.get("total_batches", 0) or 0)
    live_batch_fraction = (
        live_batch / live_total_batches if live_total_batches else 0.0
    )
    phase = str(live.get("phase", ""))
    if live_epoch > completed_epochs and total_epochs:
        if phase == "train":
            within_epoch = 0.9 * live_batch_fraction
        elif phase == "val":
            within_epoch = 0.9 + 0.1 * live_batch_fraction
        else:
            within_epoch = live_batch_fraction
        fraction = min(
            (live_epoch - 1 + within_epoch) / total_epochs,
            1.0,
        )
    else:
        fraction = completed_epochs / total_epochs if total_epochs else 0.0
    if smoke_complete:
        fraction = 1.0
    if complete:
        state = "complete"
    elif smoke_complete:
        state = "smoke_complete"
    elif run_state.get("status") == "failed" or launcher_state.get("status") == "failed":
        state = "failed"
    elif not config:
        state = "waiting_for_run_config"
    elif stale_seconds is not None and stale_seconds > stale_threshold:
        state = "stale_or_paused"
    elif completed_epochs or live.get("status") in {"running", "phase_complete"}:
        state = "training"
    else:
        state = "initializing"

    displayed_epoch = max(completed_epochs, live_epoch)
    return {
        "run_dir": str(run_dir),
        "state": state,
        "model": config.get("model"),
        "gpu_configured": config.get("gpu"),
        "epoch": displayed_epoch,
        "completed_epochs": completed_epochs,
        "total_epochs": total_epochs,
        "configured_epochs": configured_epochs,
        "progress_fraction": fraction,
        "progress_percent": 100.0 * fraction,
        "elapsed_seconds": elapsed_seconds,
        "recent_mean_epoch_seconds": recent_epoch_seconds,
        "eta_seconds": 0.0 if complete or smoke_complete else eta_seconds,
        "latest": latest,
        "live": live,
        "best": best,
        "patience": patience,
        "epochs_without_improvement": epochs_without_improvement,
        "updated_at": (
            datetime.fromtimestamp(newest_time).isoformat(timespec="seconds")
            if newest_time is not None
            else None
        ),
        "stale_seconds": stale_seconds,
        "final": final,
        "smoke": smoke,
        "run_state": run_state,
        "launcher_state": launcher_state,
        "configured_args": config.get("args", {}),
        "split_sizes": config.get("split_sizes", {}),
        "parameter_count": config.get("parameter_count"),
        "gpu_live": gpu_snapshot(),
    }


def metric_percent(row: dict[str, Any], key: str) -> str:
    return f"{100.0 * float(row[key]):.2f}%"


def render_text(snapshot: dict[str, Any]) -> str:
    lines = [
        "WHU-STree Training Monitor",
        f"Run      : {snapshot['run_dir']}",
        f"State    : {snapshot['state']}",
    ]
    if snapshot["model"]:
        lines.append(f"Model    : {snapshot['model']}")
    args = snapshot["configured_args"]
    if args:
        lines.append(
            "Setup    : "
            f"batch={args.get('batch_size')}  "
            f"eval_batch={args.get('eval_batch_size')}  "
            f"lr={args.get('learning_rate')}  "
            f"patience={args.get('patience')}  "
            f"seed={args.get('seed')}"
        )
    if snapshot["parameter_count"]:
        split_sizes = snapshot["split_sizes"]
        lines.append(
            f"Params   : {int(snapshot['parameter_count']):,}  "
            f"data={split_sizes.get('train', '?')}/"
            f"{split_sizes.get('val', '?')}/"
            f"{split_sizes.get('test', '?')} train/val/test"
        )
    lines.extend(
        [
            "",
            (
                f"Progress : {progress_bar(float(snapshot['progress_fraction']))} "
                f"{snapshot['epoch']}/{snapshot['total_epochs']} "
                f"({snapshot['progress_percent']:.2f}%)"
            ),
            f"Elapsed  : {format_duration(snapshot['elapsed_seconds'])}",
            f"Mean/ep  : {format_duration(snapshot['recent_mean_epoch_seconds'])}",
            f"ETA      : {format_duration(snapshot['eta_seconds'])}",
            f"Updated  : {snapshot['updated_at'] or 'not yet'}",
        ]
    )
    if snapshot["stale_seconds"] is not None:
        lines.append(f"Age      : {format_duration(snapshot['stale_seconds'])}")

    live = snapshot["live"]
    if live and live.get("status") in {"running", "phase_complete"}:
        batch = int(live.get("batch", 0) or 0)
        total_batches = int(live.get("total_batches", 0) or 0)
        batch_fraction = batch / total_batches if total_batches else 0.0
        lines.extend(
            [
                "",
                (
                    f"In epoch : {live.get('phase', '-')}  "
                    f"{progress_bar(batch_fraction, 24)} "
                    f"{batch}/{total_batches}"
                ),
                (
                    f"Phase ETA: "
                    f"{format_duration(live.get('phase_eta_seconds'))}"
                ),
            ]
        )
        if live.get("loss") is not None:
            lines.append(
                f"Current  : loss={float(live['loss']):.4f}  "
                f"acc={100*float(live['accuracy']):.2f}%  "
                f"macro_f1={100*float(live['macro_f1']):.2f}%"
            )
        if live.get("peak_allocated_mib") is not None:
            lines.append(
                f"Peak VRAM: {float(live['peak_allocated_mib']):.0f} MiB"
            )

    latest = snapshot["latest"]
    if latest:
        lines.extend(
            [
                "",
                (
                    f"Latest   : epoch {int(latest['epoch'])}  "
                    f"lr={float(latest['learning_rate']):.7f}"
                ),
                (
                    f"Train    : loss={float(latest['train_loss']):.4f}  "
                    f"acc={metric_percent(latest, 'train_accuracy')}  "
                    f"macro_f1={metric_percent(latest, 'train_macro_f1')}"
                ),
                (
                    f"Val      : loss={float(latest['val_loss']):.4f}  "
                    f"acc={metric_percent(latest, 'val_accuracy')}  "
                    f"macro_f1={metric_percent(latest, 'val_macro_f1')}"
                ),
            ]
        )
    best = snapshot["best"]
    if best:
        lines.append(
            f"Best val : epoch {int(best['epoch'])}  "
            f"acc={metric_percent(best, 'val_accuracy')}  "
            f"macro_f1={metric_percent(best, 'val_macro_f1')}"
        )
    if snapshot["patience"]:
        lines.append(
            f"No improve: {snapshot['epochs_without_improvement']}/"
            f"{snapshot['patience']} epoch(s)"
        )

    gpu = snapshot["gpu_live"]
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

    final = snapshot["final"]
    if final and final.get("metrics"):
        val = final["metrics"]["val"]
        test = final["metrics"]["test"]
        if final.get("note"):
            lines.extend(["", f"Metrics  : {final['note']}"])
        lines.extend(
            [
                (
                    "Stopped  : early stopping"
                    if snapshot["epoch"] < snapshot["configured_epochs"]
                    else "Stopped  : configured epoch limit"
                ),
                f"Final model checkpoint: best epoch {int(final['best_epoch'])}",
                (
                    f"Final val : acc={100*float(val['accuracy']):.2f}%  "
                    f"macro_f1={100*float(val['macro_f1']):.2f}%"
                ),
                (
                    f"Final test: acc={100*float(test['accuracy']):.2f}%  "
                    f"macro_f1={100*float(test['macro_f1']):.2f}%"
                ),
            ]
        )
    if snapshot["state"] == "failed":
        failure = snapshot["launcher_state"] or snapshot["run_state"]
        lines.extend(
            [
                "",
                f"Failure  : {failure.get('message', 'training process failed')}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    run_dir = args.run_dir.resolve()
    while True:
        snapshot = build_snapshot(run_dir)
        if not args.json:
            os.system("cls" if os.name == "nt" else "clear")
            print(render_text(snapshot), flush=True)
            if not args.once and not (
                args.exit_on_complete and snapshot["state"] == "complete"
            ):
                print("\nPress Ctrl+C to stop monitoring; training continues.", flush=True)
        else:
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)

        if args.once or (args.exit_on_complete and snapshot["state"] == "complete"):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
