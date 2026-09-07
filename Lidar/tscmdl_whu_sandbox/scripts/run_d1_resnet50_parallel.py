"""Run incomplete D1 clean-image ResNet50 seeds with a bounded queue."""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = (
    PROJECT_ROOT
    / "lidar data"
    / "whu"
    / "derived"
    / "tscmdl"
    / "d1_resnet50_clean_repeats"
    / "20260817_protocol_v1"
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # On Windows, a short-lived reader or security scanner can transiently
        # block ReplaceFile/MoveFileEx. Retry the atomic swap instead of losing
        # the queue supervisor and orphaning active training processes.
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.5))
    finally:
        if temporary.exists():
            temporary.unlink()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def write_state(path: Path, **values: object) -> None:
    atomic_json(path, {"updated_at": timestamp(), **values})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def training_command(
    protocol: dict[str, Any], suite_root: Path, seed: int
) -> tuple[list[str], Path]:
    dataset_root = PROJECT_ROOT / str(protocol["dataset_root"])
    run_dir = suite_root / "runs" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    hyper = protocol["hyperparameters"]
    execution = {
        "amp": bool(hyper["amp"]),
        "channels_last": bool(hyper["channels_last"]),
        "fused_sgd": bool(hyper["fused_sgd"]),
        "imagenet_pretrained": bool(hyper["imagenet_pretrained"]),
    }
    seed_overrides = protocol.get("per_seed_execution_overrides", {}).get(
        str(seed), {}
    )
    for name in execution:
        if name in seed_overrides:
            execution[name] = bool(seed_overrides[name])
    command = [
        sys.executable,
        str(SANDBOX_ROOT / "scripts" / "train_b3_resnet50.py"),
        "--dataset-root", str(dataset_root),
        "--output-dir", str(run_dir),
        "--quality-review", str(dataset_root / "quality_reviews_for_training.json"),
        "--epochs", str(hyper["epochs"]),
        "--batch-size", str(hyper["train_batch_size"]),
        "--eval-batch-size", str(hyper["eval_batch_size"]),
        "--learning-rate", str(hyper["learning_rate"]),
        "--min-learning-rate", str(hyper["minimum_learning_rate"]),
        "--momentum", str(hyper["momentum"]),
        "--weight-decay", str(hyper["weight_decay"]),
        "--patience", str(hyper["patience"]),
        "--seed", str(seed),
        "--workers", str(hyper["workers"]),
        "--progress-every", "100",
        "--pause-request", str(run_dir / "pause_request.json"),
        "--label-smoothing", str(hyper["label_smoothing"]),
        "--balanced-sampler",
        "--amp" if execution["amp"] else "--no-amp",
        "--channels-last" if execution["channels_last"] else "--no-channels-last",
        "--fused-sgd" if execution["fused_sgd"] else "--no-fused-sgd",
        (
            "--imagenet-pretrained"
            if execution["imagenet_pretrained"]
            else "--no-imagenet-pretrained"
        ),
    ]
    if (run_dir / "last.pt").is_file():
        command.append("--resume")
    return command, run_dir


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    protocol_path = suite_root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = read_json(protocol_path)
    seeds = [int(seed) for seed in protocol["seeds"]]
    if seeds != [20260728, 20260729, 20260730]:
        raise ValueError(f"Unexpected D1 seeds: {seeds}")
    hyper = protocol["hyperparameters"]
    if int(hyper["train_batch_size"]) != 8 or not hyper["balanced_sampler"]:
        raise ValueError("Parallel runner refuses to alter the frozen scientific protocol")
    commands = {
        seed: training_command(protocol, suite_root, seed)[0] for seed in seeds
    }
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "suite_root": str(suite_root),
                    "seeds": seeds,
                    "commands": commands,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    lock_path = suite_root / "parallel_runner.lock"
    lock_stream = lock_path.open("a+b")
    lock_stream.seek(0)
    if lock_path.stat().st_size == 0:
        lock_stream.write(b"0")
        lock_stream.flush()
        lock_stream.seek(0)
    try:
        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        raise RuntimeError("D1 parallel runner is already active") from error

    state_path = suite_root / "suite_state.json"
    existing_state = read_json(state_path) if state_path.is_file() else {}
    if existing_state.get("status") == "running":
        raise RuntimeError("Sequential runner is still active; pause it safely first")
    max_parallel = 2
    processes: dict[int, subprocess.Popen[bytes]] = {}
    logs: dict[int, Any] = {}
    completed: list[int] = []
    failed: list[int] = []
    paused: list[int] = []
    pending: list[int] = []
    for seed in seeds:
        run_dir = suite_root / "runs" / f"seed_{seed}"
        final_path = run_dir / "final_metrics.json"
        if final_path.is_file() and read_json(final_path).get("status") == "complete":
            completed.append(seed)
            continue
        pending.append(seed)

    def start_seed(seed: int) -> None:
        command, run_dir = training_command(protocol, suite_root, seed)
        log = (run_dir / "parallel_launcher_stdout.log").open(
            "a", encoding="utf-8", buffering=1
        )
        log.write(
            f"[{timestamp()}] parallel start resume={'--resume' in command} "
            f"command={json.dumps(command)}\n"
        )
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        logs[seed] = log
        processes[seed] = process
        write_state(
            run_dir / "launcher_state.json",
            status="running",
            mode="parallel",
            launcher_pid=os.getpid(),
            training_pid=process.pid,
            seed=seed,
            resumed="--resume" in command,
            command=command,
        )

    def update_suite(status: str) -> None:
        write_state(
            state_path,
            status=status,
            runner_pid=os.getpid(),
            execution_mode="parallel_max_2",
            max_parallel=max_parallel,
            seeds=seeds,
            current_seed=None,
            current_run_dir=None,
            current_seeds=sorted(processes),
            pending_seeds=list(pending),
            completed_seeds=sorted(completed),
            paused_seeds=sorted(paused),
            failed_seeds=sorted(failed),
        )

    stop_queue = False
    while pending and len(processes) < max_parallel:
        start_seed(pending.pop(0))
    update_suite("running_parallel")
    while processes or pending:
        for seed, process in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            run_dir = suite_root / "runs" / f"seed_{seed}"
            run_state = (
                read_json(run_dir / "run_state.json")
                if (run_dir / "run_state.json").is_file()
                else {}
            )
            if return_code != 0:
                failed.append(seed)
                launcher_status = "failed"
                stop_queue = True
            elif run_state.get("status") == "complete":
                completed.append(seed)
                launcher_status = "complete"
            elif run_state.get("status") == "paused":
                paused.append(seed)
                launcher_status = "paused"
                stop_queue = True
            else:
                failed.append(seed)
                launcher_status = "failed_incomplete_state"
                stop_queue = True
            write_state(
                run_dir / "launcher_state.json",
                status=launcher_status,
                mode="parallel",
                seed=seed,
                return_code=return_code,
                run_state=run_state,
            )
            logs[seed].write(
                f"[{timestamp()}] exited rc={return_code} status={launcher_status}\n"
            )
            logs[seed].close()
            del logs[seed]
            del processes[seed]
        while pending and len(processes) < max_parallel and not stop_queue:
            start_seed(pending.pop(0))
        update_suite(
            "running_parallel" if processes else "finishing"
        )
        if not processes and (stop_queue or not pending):
            break
        if processes:
            time.sleep(5.0)

    final_status = (
        "failed" if failed else "paused" if paused else "complete"
    )
    update_suite(final_status)


if __name__ == "__main__":
    main()
