"""Build one self-contained Windows folder for migrating the active C2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import conda_pack
import torch


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_DATASET = DERIVED_ROOT / "c1_full_shared_dataset"
DEFAULT_CACHE = DERIVED_ROOT / "c2a_ptv2" / "knn_memmap_k8"
DEFAULT_RUN = DERIVED_ROOT / "c2_ptv2" / "run_batch16_seed20260731"
DEFAULT_ENV = PROJECT_ROOT / "envs" / "whu-tscmdl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--environment-prefix", type=Path, default=DEFAULT_ENV)
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


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(
            f"Portable output already exists; choose a new path: {output}"
        )
    output.mkdir(parents=True)

    template = SANDBOX_ROOT / "portable_c2_template"
    shutil.copytree(
        template,
        output,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    app_scripts = output / "app" / "scripts"
    app_package = output / "app" / "src" / "tscmdl_whu"
    app_scripts.mkdir(parents=True, exist_ok=True)
    for name in ("train_b5_ptv2.py", "monitor_training.py"):
        shutil.copy2(SANDBOX_ROOT / "scripts" / name, app_scripts / name)
    for name in ("knn_cache.py", "ptv2.py", "training.py"):
        shutil.copy2(
            SANDBOX_ROOT / "src" / "tscmdl_whu" / name,
            app_package / name,
        )

    dataset_root = args.dataset_root.resolve()
    dataset_output = output / "input" / "dataset"
    dataset_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_root / "manifest.json", dataset_output / "manifest.json")
    shutil.copy2(dataset_root / "classes.json", dataset_output / "classes.json")
    manifest = json.loads(
        (dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    point_paths = sorted(
        {str(record["point_path"]) for record in manifest["records"]}
    )
    for index, relative in enumerate(point_paths, start=1):
        link_or_copy(dataset_root / relative, dataset_output / relative)
        if index % 1000 == 0 or index == len(point_paths):
            print(
                f"\rPoint assets: {index}/{len(point_paths)}",
                end="",
                flush=True,
            )
    print(flush=True)

    cache_root = args.cache_root.resolve()
    cache_output = output / "input" / "knn_cache"
    cache_output.mkdir(parents=True, exist_ok=True)
    for name in (
        "index.json",
        "train_reference_index.npy",
        "val_reference_index.npy",
        "test_reference_index.npy",
    ):
        link_or_copy(cache_root / name, cache_output / name)

    run_dir = args.run_dir.resolve()
    checkpoint_output = output / "input" / "checkpoint"
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    last_destination = checkpoint_output / "last.pt"
    best_destination = checkpoint_output / "best.pt"
    link_or_copy(run_dir / "last.pt", last_destination)
    link_or_copy(run_dir / "best.pt", best_destination)
    last = torch.load(last_destination, map_location="cpu", weights_only=False)
    best = torch.load(best_destination, map_location="cpu", weights_only=False)
    if int(best["epoch"]) != int(last["best_epoch"]):
        raise ValueError(
            "Best checkpoint changed while snapshotting; run the builder again"
        )
    history = last["history"]
    atomic_json(checkpoint_output / "training_history.json", history)
    write_history_csv(checkpoint_output / "training_history.csv", history)
    for name in ("run_config.json", "train.log"):
        shutil.copy2(run_dir / name, checkpoint_output / name)
    snapshot = {
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source_run": str(run_dir),
        "checkpoint_epoch": int(last["epoch"]),
        "best_epoch": int(last["best_epoch"]),
        "history_rows": len(history),
        "last_sha256": sha256_file(last_destination),
        "best_sha256": sha256_file(best_destination),
        "continuation_policy": "exact batch-16 resume",
    }
    atomic_json(checkpoint_output / "migration_snapshot.json", snapshot)

    runtime_dir = output / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    archive = runtime_dir / "whu-tscmdl-windows-x64.tar.gz"
    print("Packing the verified whu-tscmdl environment...", flush=True)
    conda_pack.pack(
        prefix=str(args.environment_prefix.resolve()),
        output=str(archive),
        format="tar.gz",
        compress_level=1,
        force=True,
    )

    mutable_roots = {
        str((output / "output").resolve()).lower(),
        str((output / "diagnostics").resolve()).lower(),
        str((output / "runtime" / "env").resolve()).lower(),
    }
    entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        resolved = str(path.resolve()).lower()
        if any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in mutable_roots
        ):
            continue
        if path.name == "transfer_manifest.json":
            continue
        entries.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if len(entries) % 500 == 0:
            print(f"\rHashing bundle files: {len(entries)}", end="", flush=True)
    print(flush=True)
    transfer_manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "platform": "Windows x86_64",
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "checkpoint": snapshot,
        "files": entries,
    }
    atomic_json(output / "transfer_manifest.json", transfer_manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(output),
                "file_count": len(entries),
                "total_gib": round(
                    transfer_manifest["total_bytes"] / (1024**3),
                    3,
                ),
                "checkpoint_epoch": snapshot["checkpoint_epoch"],
                "best_epoch": snapshot["best_epoch"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
