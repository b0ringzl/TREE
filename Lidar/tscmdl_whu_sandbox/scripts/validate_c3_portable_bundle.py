"""Run fast structural validation of a C3 portable training folder."""

from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.bundle_root.resolve()
    errors: list[str] = []
    required = [
        "START_C3_TRAINING.bat",
        "MONITOR_C3.bat",
        "CHECK_DEVICE.bat",
        "PACK_C3_RESULTS.bat",
        "使用说明.md",
        "c3_config.json",
        "transfer_manifest.json",
        "runtime/whu-tscmdl-windows-x64.tar.gz",
        "input/dataset/manifest.json",
        "input/dataset/classes.json",
        "input/knn_cache/index.json",
        "app/scripts/train_b5_ptv2.py",
        "app/scripts/monitor_training.py",
        "app/scripts/monitor_c3.py",
        "app/scripts/summarize_c3_results.py",
        "tools/device_probe_c3.py",
        "tools/verify_bundle.py",
        "tools/prepare_runtime.ps1",
        "tools/run_c3.ps1",
        "tools/start_c3_and_monitor.ps1",
        "tools/pack_results.ps1",
    ]
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    config = json.loads((root / "c3_config.json").read_text(encoding="utf-8"))
    expected_config = {
        "seeds": [20260728, 20260729, 20260730],
        "batch_size": 48,
        "eval_batch_size": 64,
        "epochs": 120,
        "patience": 20,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(f"unexpected C3 config {key}: {config.get(key)!r}")

    dataset = json.loads(
        (root / "input" / "dataset" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    classes = json.loads(
        (root / "input" / "dataset" / "classes.json").read_text(
            encoding="utf-8"
        )
    )
    if int(dataset["summary"]["sample_count"]) != 17134:
        errors.append("dataset sample count is not 17,134")
    if dataset["summary"]["split_histogram"] != {
        "train": 13116,
        "val": 570,
        "test": 3448,
    }:
        errors.append("dataset split sizes changed")
    if len(classes) != 19:
        errors.append("class count is not 19")

    manifest = json.loads(
        (root / "transfer_manifest.json").read_text(encoding="utf-8")
    )
    manifest_paths = [str(entry["path"]) for entry in manifest["files"]]
    if len(manifest_paths) != len(set(manifest_paths)):
        errors.append("transfer manifest contains duplicate paths")
    if any(path.startswith("input/checkpoint/") for path in manifest_paths):
        errors.append("C3 bundle must not include a C2 checkpoint")
    if manifest.get("stage") != "C3":
        errors.append("transfer manifest stage is not C3")
    missing_manifest_files = []
    size_mismatches = []
    for entry in manifest["files"]:
        path = root / str(entry["path"])
        if not path.is_file():
            missing_manifest_files.append(str(entry["path"]))
        elif path.stat().st_size != int(entry["bytes"]):
            size_mismatches.append(str(entry["path"]))
    if missing_manifest_files:
        errors.append(f"manifest files missing: {len(missing_manifest_files)}")
    if size_mismatches:
        errors.append(f"manifest size mismatches: {len(size_mismatches)}")

    compile_errors = []
    compile_dir = root / "diagnostics" / "py_compile"
    compile_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        root / "app" / "scripts" / "train_b5_ptv2.py",
        root / "app" / "scripts" / "monitor_training.py",
        root / "app" / "scripts" / "monitor_c3.py",
        root / "app" / "scripts" / "summarize_c3_results.py",
        root / "tools" / "device_probe_c3.py",
        root / "tools" / "verify_bundle.py",
    ]:
        try:
            py_compile.compile(
                str(path),
                cfile=str(compile_dir / f"{path.stem}.pyc"),
                doraise=True,
            )
        except py_compile.PyCompileError as exc:
            compile_errors.append(f"{path.name}: {exc}")
    errors.extend(compile_errors)

    powershell_errors = []
    for path in sorted((root / "tools").glob("*.ps1")):
        command = (
            "$ErrorActionPreference='Stop'; "
            f"[void][ScriptBlock]::Create((Get-Content -LiteralPath '{path}' -Raw)); "
            "Write-Output OK"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or "OK" not in result.stdout:
            powershell_errors.append(f"{path.name}: {result.stderr.strip()}")
    errors.extend(powershell_errors)

    report = {
        "status": "passed" if not errors else "failed",
        "validated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "root": str(root),
        "file_count": int(manifest["file_count"]),
        "logical_total_gib": round(int(manifest["total_bytes"]) / (1024**3), 3),
        "sample_count": int(dataset["summary"]["sample_count"]),
        "class_count": len(classes),
        "fixed_config": expected_config,
        "errors": errors,
    }
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "local_bundle_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
