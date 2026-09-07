"""Atomically refresh the portable bundle with the latest completed C2 epoch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = (
    PROJECT_ROOT
    / "lidar data"
    / "whu"
    / "derived"
    / "tscmdl"
    / "c2_ptv2"
    / "run_batch16_seed20260731"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_history_csv(
    path: Path,
    history: list[dict[str, object]],
) -> None:
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "epoch_seconds",
        "peak_allocated_mib",
    ]
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def link_snapshot(source: Path, destination: Path) -> None:
    temporary = Path(f"{destination}.refresh")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    bundle = args.bundle_root.resolve()
    run_dir = args.run_dir.resolve()
    checkpoint_root = bundle / "input" / "checkpoint"
    manifest_path = bundle / "transfer_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    for attempt in range(3):
        link_snapshot(run_dir / "last.pt", checkpoint_root / "last.pt")
        link_snapshot(run_dir / "best.pt", checkpoint_root / "best.pt")
        last = torch.load(
            checkpoint_root / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        best = torch.load(
            checkpoint_root / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        if int(best["epoch"]) == int(last["best_epoch"]):
            break
        if attempt == 2:
            raise ValueError("Could not obtain a coherent last/best snapshot")
        time.sleep(1.0)

    history = last["history"]
    atomic_json(checkpoint_root / "training_history.json", history)
    write_history_csv(checkpoint_root / "training_history.csv", history)
    for name in ("run_config.json", "train.log"):
        shutil.copy2(run_dir / name, checkpoint_root / name)
    snapshot = {
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source_run": str(run_dir),
        "checkpoint_epoch": int(last["epoch"]),
        "best_epoch": int(last["best_epoch"]),
        "history_rows": len(history),
        "last_sha256": sha256_file(checkpoint_root / "last.pt"),
        "best_sha256": sha256_file(checkpoint_root / "best.pt"),
        "continuation_policy": "exact batch-16 resume",
    }
    atomic_json(checkpoint_root / "migration_snapshot.json", snapshot)

    transfer = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {str(entry["path"]): entry for entry in transfer["files"]}
    for path in sorted(item for item in checkpoint_root.rglob("*") if item.is_file()):
        relative = path.relative_to(bundle).as_posix()
        entries[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    ordered = [entries[key] for key in sorted(entries)]
    transfer["files"] = ordered
    transfer["file_count"] = len(ordered)
    transfer["total_bytes"] = sum(int(entry["bytes"]) for entry in ordered)
    transfer["checkpoint"] = snapshot
    atomic_json(manifest_path, transfer)
    atomic_json(
        bundle / "diagnostics" / "integrity_report.json",
        {
            "status": "stale_after_checkpoint_refresh",
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        },
    )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
