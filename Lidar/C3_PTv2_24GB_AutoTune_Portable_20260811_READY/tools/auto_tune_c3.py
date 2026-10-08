"""Select the fastest safe fixed C3 batch size on the target 24 GB GPU."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def replace_with_retry(temporary: Path, destination: Path) -> None:
    delay_seconds = 0.01
    deadline = time.monotonic() + 10.0
    while True:
        try:
            temporary.replace(destination)
            return
        except OSError as error:
            retryable = os.name == "nt" and getattr(error, "winerror", None) in {5, 32}
            if not retryable or time.monotonic() >= deadline:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.6, 0.5)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    replace_with_retry(temporary, path)


def run_probe(
    *,
    batch_size: int,
    steps: int,
    output_dir: Path,
    config: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "app" / "scripts" / "train_b5_ptv2.py"),
        "--dataset-root",
        str(ROOT / "input" / "dataset"),
        "--knn-cache",
        str(ROOT / "input" / "knn_cache"),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        str(config["eval_batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--min-learning-rate",
        str(config["min_learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--label-smoothing",
        str(config["label_smoothing"]),
        "--patience",
        str(config["patience"]),
        "--seed",
        str(config["seeds"][0]),
        "--workers",
        str(config["workers"]),
        "--progress-every",
        str(max(1, steps)),
        "--smoke-steps",
        str(steps),
        "--amp",
    ]
    stdout_path = output_dir / "console.out.log"
    stderr_path = output_dir / "console.err.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=3600,
        )
    elapsed = time.perf_counter() - started
    result_path = output_dir / "smoke_result.json"
    if completed.returncode != 0 or not result_path.is_file():
        error_tail = ""
        if stderr_path.is_file():
            error_tail = "\n".join(
                stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            )
        return {
            "status": "failed",
            "batch_size": batch_size,
            "exit_code": completed.returncode,
            "wall_seconds": elapsed,
            "error_tail": error_tail,
        }

    smoke = json.loads(result_path.read_text(encoding="utf-8-sig"))
    train = smoke["train"]
    samples_per_second = float(train["sample_count"]) / float(
        train["elapsed_seconds"]
    )
    return {
        "status": "passed",
        "batch_size": batch_size,
        "steps": int(train["successful_optimizer_steps"]),
        "sample_count": int(train["sample_count"]),
        "train_seconds": float(train["elapsed_seconds"]),
        "samples_per_second": samples_per_second,
        "peak_allocated_mib": float(train["peak_allocated_mib"]),
        "wall_seconds": elapsed,
        "output_dir": str(output_dir),
    }


def main() -> None:
    config_path = ROOT / "c3_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    tune = config["auto_tune"]
    report_path = ROOT / "diagnostics" / "auto_tune_report.json"
    computer_name = platform.node()
    gpu_name = torch.cuda.get_device_name(0)
    total_mib = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if (
            previous.get("status") == "passed"
            and previous.get("computer_name") == computer_name
            and previous.get("gpu_name") == gpu_name
            and int(previous["selected_batch"]) == int(config["batch_size"])
        ):
            print(
                f"Auto-tuning already completed: batch={config['batch_size']} on {gpu_name}",
                flush=True,
            )
            return

    checkpoints = list((ROOT / "output" / "runs").glob("seed_*/last.pt"))
    if checkpoints:
        raise RuntimeError(
            "Refusing to retune after real C3 checkpoints exist. Restore the matching "
            "auto_tune_report.json or use a fresh package."
        )

    diagnostics = ROOT / "diagnostics" / "auto_tune"
    diagnostics.mkdir(parents=True, exist_ok=True)
    warmup_steps = int(tune["warmup_steps"])
    smoke_steps = int(tune["smoke_steps"])
    candidates = [int(value) for value in tune["candidates"]]
    max_peak_mib = total_mib * float(tune["max_peak_allocated_fraction"])

    print(
        f"Warming up {gpu_name} with {warmup_steps} optimizer steps...",
        flush=True,
    )
    warmup = run_probe(
        batch_size=min(candidates),
        steps=warmup_steps,
        output_dir=diagnostics / "warmup",
        config=config,
    )
    if warmup["status"] != "passed":
        raise RuntimeError("GPU warmup failed; review diagnostics/auto_tune/warmup.")

    results: list[dict[str, object]] = []
    for batch_size in candidates:
        print(
            f"Testing batch={batch_size} for {smoke_steps} real optimizer steps...",
            flush=True,
        )
        result = run_probe(
            batch_size=batch_size,
            steps=smoke_steps,
            output_dir=diagnostics / f"batch_{batch_size}",
            config=config,
        )
        if result["status"] == "passed":
            result["eligible"] = float(result["peak_allocated_mib"]) <= max_peak_mib
            result["headroom_rule_mib"] = max_peak_mib
            print(
                "  throughput={:.2f} samples/s, peak={:.0f} MiB, eligible={}".format(
                    float(result["samples_per_second"]),
                    float(result["peak_allocated_mib"]),
                    result["eligible"],
                ),
                flush=True,
            )
        else:
            result["eligible"] = False
            print(f"  batch={batch_size} failed and will not be selected.", flush=True)
        results.append(result)

    eligible = [
        result
        for result in results
        if result["status"] == "passed" and bool(result["eligible"])
    ]
    if not eligible:
        raise RuntimeError("No safe auto-tuning candidate passed.")
    selected = max(eligible, key=lambda item: float(item["samples_per_second"]))
    selected_batch = int(selected["batch_size"])
    selected_at = datetime.now().astimezone().isoformat(timespec="seconds")

    config["batch_size"] = selected_batch
    config["auto_tune"]["selected_batch"] = selected_batch
    config["auto_tune"]["selected_at"] = selected_at
    report = {
        "schema_version": 1,
        "status": "passed",
        "selected_at": selected_at,
        "computer_name": computer_name,
        "gpu_name": gpu_name,
        "total_vram_mib": total_mib,
        "max_peak_allocated_mib": max_peak_mib,
        "candidates": candidates,
        "results": results,
        "selected_batch": selected_batch,
        "selection_metric": tune["selection_metric"],
    }
    atomic_json(config_path, config)
    atomic_json(report_path, report)
    print(
        "Selected fixed C3 batch={} at {:.2f} samples/s.".format(
            selected_batch, float(selected["samples_per_second"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
