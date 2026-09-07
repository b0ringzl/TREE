"""Run the frozen D1 four-class PTv2 protocol sequentially and resumably."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
TRAINER_PATH = SCRIPT_PATH.with_name("train_b5_ptv2.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    deadline = time.monotonic() + 10.0
    while True:
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if (
                os.name != "nt"
                or getattr(error, "winerror", None) not in {5, 32}
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(0.05)


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_state(path: Path, **values: Any) -> None:
    atomic_json(path, {"updated_at": now(), "runner_pid": os.getpid(), **values})


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    protocol_path = suite_root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise ValueError("D1 PTv2 protocol is not frozen")
    if protocol.get("stage") != "D1-four-class-PTv2-three-seed":
        raise ValueError("Unexpected D1 PTv2 protocol stage")

    for name, binding in protocol["source_bindings"].items():
        path = resolve_project_path(str(binding["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Missing source binding {name}: {path}")
        actual = sha256_file(path)
        if actual != str(binding["sha256"]):
            raise ValueError(f"Source binding hash changed for {name}: {path}")

    dataset_root = resolve_project_path(str(protocol["dataset_root"]))
    cache_root = resolve_project_path(str(protocol["knn_cache_root"]))
    hyperparameters = protocol["hyperparameters"]
    seeds = [int(seed) for seed in protocol["seeds"]]
    commands: list[list[str]] = []
    for seed in seeds:
        run_dir = suite_root / "runs" / f"seed_{seed}"
        pause_request = run_dir / "pause_request.json"
        command = [
            sys.executable,
            str(TRAINER_PATH),
            "--dataset-root",
            str(dataset_root),
            "--knn-cache",
            str(cache_root),
            "--output-dir",
            str(run_dir),
            "--epochs",
            str(hyperparameters["epochs"]),
            "--batch-size",
            str(hyperparameters["batch_size"]),
            "--eval-batch-size",
            str(hyperparameters["eval_batch_size"]),
            "--learning-rate",
            str(hyperparameters["learning_rate"]),
            "--min-learning-rate",
            str(hyperparameters["minimum_learning_rate"]),
            "--weight-decay",
            str(hyperparameters["weight_decay"]),
            "--label-smoothing",
            str(hyperparameters["label_smoothing"]),
            "--patience",
            str(hyperparameters["patience"]),
            "--seed",
            str(seed),
            "--workers",
            str(hyperparameters["workers"]),
            "--progress-every",
            str(hyperparameters["progress_every"]),
            "--pause-request",
            str(pause_request),
        ]
        command.append("--amp" if hyperparameters["amp"] else "--no-amp")
        command.append(
            "--balanced-sampler"
            if hyperparameters["balanced_sampler"]
            else "--no-balanced-sampler"
        )
        if (run_dir / "last.pt").is_file():
            command.append("--resume")
        commands.append(command)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "suite_root": str(suite_root),
                    "dataset_root": str(dataset_root),
                    "cache_root": str(cache_root),
                    "commands": commands,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    state_path = suite_root / "suite_state.json"
    completed: list[int] = []
    for seed, command in zip(seeds, commands, strict=True):
        run_dir = suite_root / "runs" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        final = read_json(run_dir / "final_metrics.json")
        if final and final.get("status") == "complete":
            completed.append(seed)
            continue
        pause_request = run_dir / "pause_request.json"
        if pause_request.is_file():
            write_state(
                state_path,
                status="paused",
                active_seed=seed,
                seeds=seeds,
                completed_seeds=completed,
                message="Existing pause request prevents launch",
            )
            return

        resumed = "--resume" in command
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stdout_path = run_dir / f"console_stdout_{timestamp}.log"
        stderr_path = run_dir / f"console_stderr_{timestamp}.log"
        write_state(
            state_path,
            status="running",
            active_seed=seed,
            seeds=seeds,
            completed_seeds=completed,
            message=f"{'Resuming' if resumed else 'Starting'} seed {seed}",
        )
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        final = read_json(run_dir / "final_metrics.json")
        run_state = read_json(run_dir / "run_state.json") or {}
        if final and final.get("status") == "complete":
            completed.append(seed)
            continue
        if run_state.get("status") == "paused":
            write_state(
                state_path,
                status="paused",
                active_seed=seed,
                seeds=seeds,
                completed_seeds=completed,
                message=f"Seed {seed} paused after a saved epoch",
            )
            return
        write_state(
            state_path,
            status="failed",
            active_seed=seed,
            seeds=seeds,
            completed_seeds=completed,
            exit_code=result.returncode,
            stderr_log=str(stderr_path),
            message=f"Seed {seed} exited without complete metrics",
        )
        raise SystemExit(result.returncode or 1)

    write_state(
        state_path,
        status="complete",
        active_seed=0,
        seeds=seeds,
        completed_seeds=completed,
        message="All D1 PTv2 seeds completed",
    )


if __name__ == "__main__":
    main()
