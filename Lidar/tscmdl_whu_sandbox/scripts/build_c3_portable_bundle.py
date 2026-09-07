"""Build the self-contained C3 three-seed portable training folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = PROJECT_ROOT / "Lidar" / "C2_PTv2_Portable_20260731_READY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-c2-bundle", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, required=True)
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


def inherit_path(relative: str) -> bool:
    prefixes = (
        "app/src/",
        "input/dataset/",
        "input/knn_cache/",
        "runtime/whu-tscmdl-windows-x64.tar.gz",
    )
    exact = {
        "app/scripts/train_b5_ptv2.py",
        "app/scripts/monitor_training.py",
    }
    return relative in exact or relative.startswith(prefixes)


def main() -> None:
    args = parse_args()
    source = args.source_c2_bundle.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    source_manifest_path = source / "transfer_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_entries = {
        str(entry["path"]): entry for entry in source_manifest["files"]
    }
    inherited: dict[str, dict[str, object]] = {}
    hardlinks = 0
    copies = 0
    selected = [
        relative for relative in sorted(source_entries) if inherit_path(relative)
    ]
    for index, relative in enumerate(selected, start=1):
        source_path = source / relative
        destination = output / relative
        method = link_or_copy(source_path, destination)
        if method == "hardlink":
            hardlinks += 1
        else:
            copies += 1
        inherited[relative] = source_entries[relative]
        if index % 1000 == 0 or index == len(selected):
            print(
                f"\rInherited files: {index}/{len(selected)}",
                end="",
                flush=True,
            )
    print(flush=True)

    template = SANDBOX_ROOT / "portable_c3_template"
    shutil.copytree(
        template,
        output,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    app_scripts = output / "app" / "scripts"
    tools = output / "tools"
    app_scripts.mkdir(parents=True, exist_ok=True)
    tools.mkdir(parents=True, exist_ok=True)
    for name in ("monitor_c3.py", "summarize_c3_results.py"):
        shutil.copy2(SANDBOX_ROOT / "scripts" / name, app_scripts / name)
    shutil.copy2(
        SANDBOX_ROOT / "scripts" / "device_probe_c3.py",
        tools / "device_probe_c3.py",
    )
    shutil.copy2(source / "tools" / "verify_bundle.py", tools / "verify_bundle.py")

    config = {
        "schema_version": 1,
        "stage": "C3",
        "purpose": "fixed-configuration three-seed PTv2 repeat",
        "seeds": [20260728, 20260729, 20260730],
        "epochs": 120,
        "batch_size": 48,
        "eval_batch_size": 64,
        "learning_rate": 0.001,
        "min_learning_rate": 0.00001,
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
        "patience": 20,
        "workers": 0,
        "progress_every": 10,
        "amp": True,
        "checkpoint_policy": "fresh start per seed; exact same-seed resume only",
        "test_policy": "evaluate locked official test split after best validation selection",
        "historical_c2_policy": "C2 seed 20260731 remains a non-repeat historical baseline",
    }
    atomic_json(output / "c3_config.json", config)
    (output / "output").mkdir(parents=True, exist_ok=True)
    (output / "diagnostics").mkdir(parents=True, exist_ok=True)

    mutable_roots = {
        str((output / "output").resolve()).lower(),
        str((output / "diagnostics").resolve()).lower(),
        str((output / "runtime" / "env").resolve()).lower(),
    }
    entries = []
    reused_hashes = 0
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        resolved = str(path.resolve()).lower()
        if any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in mutable_roots
        ):
            continue
        if path.name == "transfer_manifest.json":
            continue
        relative = path.relative_to(output).as_posix()
        inherited_entry = inherited.get(relative)
        if inherited_entry and path.stat().st_size == int(inherited_entry["bytes"]):
            sha256 = str(inherited_entry["sha256"])
            reused_hashes += 1
        else:
            sha256 = sha256_file(path)
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256,
            }
        )

    transfer_manifest = {
        "schema_version": 1,
        "stage": "C3",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": "Windows x86_64 with NVIDIA CUDA and 24 GB-class VRAM",
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "source_c2_manifest_sha256": sha256_file(source_manifest_path),
        "storage_method": {
            "hardlinks": hardlinks,
            "copies": copies,
            "note": "Explorer/robocopy USB transfer materializes normal files.",
        },
        "fixed_config": config,
        "files": entries,
    }
    atomic_json(output / "transfer_manifest.json", transfer_manifest)
    builder_report = {
        "status": "complete",
        "output_root": str(output),
        "file_count": len(entries),
        "logical_total_gib": round(transfer_manifest["total_bytes"] / (1024**3), 3),
        "hardlinks": hardlinks,
        "copies": copies,
        "reused_hashes": reused_hashes,
    }
    atomic_json(output / "diagnostics" / "builder_report.json", builder_report)
    print(json.dumps(builder_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
