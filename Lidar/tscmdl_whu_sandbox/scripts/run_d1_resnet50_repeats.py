"""Run the frozen D1 clean-image ResNet50 protocol for three seeds in sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_DATASET = (
    DERIVED_ROOT / "d1_four_class_training_package" / "20260817_165127"
)
DEFAULT_SUITE = (
    DERIVED_ROOT
    / "d1_resnet50_clean_repeats"
    / "20260817_protocol_v1"
)
DEFAULT_SEEDS = (20260728, 20260729, 20260730)
EXPECTED_WEIGHT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
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
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    suite_root = args.suite_root.resolve()
    validation_path = dataset_root / "validation.json"
    manifest_path = dataset_root / "manifest.json"
    quality_review_path = dataset_root / "quality_reviews_for_training.json"
    train_script = SANDBOX_ROOT / "scripts" / "train_b3_resnet50.py"
    monitor_script = SANDBOX_ROOT / "scripts" / "monitor_d1_resnet50_repeats.py"
    weight_path = (
        Path.home()
        / ".cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "resnet50-11ad3fa6.pth"
    )
    required = [
        validation_path,
        manifest_path,
        quality_review_path,
        train_script,
        monitor_script,
        weight_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required D1 training inputs are missing: {missing}")
    validation = read_json(validation_path)
    if validation.get("status") not in {"passed", "passed_with_warnings"}:
        raise ValueError("D1 package validation has not passed")
    if validation.get("errors"):
        raise ValueError("D1 package validation contains errors")
    manifest = read_json(manifest_path)
    split_sizes = manifest.get("summary", {}).get("split_sizes", {})
    if split_sizes != {"test": 151, "train": 6484, "val": 90}:
        raise ValueError(f"Unexpected frozen D1 split sizes: {split_sizes}")
    if sha256_file(weight_path) != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Cached official ResNet50 weight hash mismatch")
    return {
        "schema_version": 1,
        "stage": "D1-clean-image-ResNet50-three-seed",
        "status": "frozen",
        "frozen_at": timestamp(),
        "dataset_root": relative(dataset_root),
        "suite_root": relative(suite_root),
        "seeds": [int(seed) for seed in args.seeds],
        "execution_order": "sequential",
        "split_sizes": split_sizes,
        "model": "torchvision ResNet50 with four-class linear head",
        "class_order": [
            "Cinnamomum camphora",
            "Lagerstroemia indica",
            "Magnolia grandiflora",
            "Other",
        ],
        "hyperparameters": {
            "epochs": int(args.epochs),
            "train_batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "optimizer": "fused SGD",
            "learning_rate": 0.1,
            "minimum_learning_rate": 0.0,
            "momentum": 0.9,
            "weight_decay": 0.0002,
            "scheduler": "CosineAnnealingLR",
            "loss": "CrossEntropyLoss",
            "label_smoothing": 0.0,
            "patience": int(args.epochs),
            "balanced_sampler": True,
            "workers": int(args.workers),
            "amp": True,
            "channels_last": True,
            "fused_sgd": True,
            "imagenet_pretrained": True,
        },
        "selection_policy": (
            "best checkpoint by validation macro-F1, then validation accuracy"
        ),
        "test_policy": (
            "test split is first loaded after validation-selected training completes"
        ),
        "pause_policy": "safe pause only after an epoch checkpoint is written",
        "source_bindings": {
            "manifest": {
                "path": relative(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "package_validation": {
                "path": relative(validation_path),
                "sha256": sha256_file(validation_path),
            },
            "quality_review": {
                "path": relative(quality_review_path),
                "sha256": sha256_file(quality_review_path),
            },
            "train_script": {
                "path": relative(train_script),
                "sha256": sha256_file(train_script),
            },
            "monitor_script": {
                "path": relative(monitor_script),
                "sha256": sha256_file(monitor_script),
            },
            "imagenet_weight": {
                "path": str(weight_path),
                "sha256": EXPECTED_WEIGHT_SHA256,
            },
        },
    }


def write_state(path: Path, **values: object) -> None:
    atomic_json(path, {"updated_at": timestamp(), **values})


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("Exactly three unique seeds are required")
    frozen = protocol(args)
    if args.check_only:
        print(json.dumps(frozen, ensure_ascii=False, indent=2))
        return

    dataset_root = args.dataset_root.resolve()
    suite_root = args.suite_root.resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    lock_path = suite_root / "runner.lock"
    lock_stream = lock_path.open("a+b")
    lock_stream.seek(0)
    if lock_stream.tell() == 0:
        lock_stream.write(b"0")
        lock_stream.flush()
        lock_stream.seek(0)
    try:
        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        raise RuntimeError("D1 three-seed runner is already active") from error

    protocol_path = suite_root / "protocol.json"
    if protocol_path.is_file():
        existing = read_json(protocol_path)
        comparable_keys = (
            "dataset_root",
            "seeds",
            "split_sizes",
            "hyperparameters",
            "selection_policy",
            "test_policy",
        )
        if any(existing.get(key) != frozen.get(key) for key in comparable_keys):
            raise ValueError("Existing suite protocol differs from requested protocol")
        frozen = existing
    else:
        atomic_json(protocol_path, frozen)

    state_path = suite_root / "suite_state.json"
    log_path = suite_root / "suite_launcher.log"
    write_state(
        state_path,
        status="running",
        runner_pid=os.getpid(),
        seeds=frozen["seeds"],
        completed_seeds=[],
        current_seed=None,
        current_run_dir=None,
    )
    completed: list[int] = []
    train_script = SANDBOX_ROOT / "scripts" / "train_b3_resnet50.py"
    quality_review = dataset_root / "quality_reviews_for_training.json"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{timestamp()}] runner pid={os.getpid()} protocol={protocol_path}\n")
        for seed in frozen["seeds"]:
            seed = int(seed)
            run_dir = suite_root / "runs" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            final_path = run_dir / "final_metrics.json"
            if final_path.is_file() and read_json(final_path).get("status") == "complete":
                completed.append(seed)
                log.write(f"[{timestamp()}] seed={seed} already complete; skipped\n")
                continue
            pause_path = run_dir / "pause_request.json"
            command = [
                sys.executable,
                str(train_script),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(run_dir),
                "--quality-review",
                str(quality_review),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--learning-rate",
                "0.1",
                "--min-learning-rate",
                "0.0",
                "--momentum",
                "0.9",
                "--weight-decay",
                "0.0002",
                "--patience",
                str(args.epochs),
                "--seed",
                str(seed),
                "--workers",
                str(args.workers),
                "--progress-every",
                "100",
                "--pause-request",
                str(pause_path),
                "--label-smoothing",
                "0.0",
                "--balanced-sampler",
                "--amp",
                "--channels-last",
                "--fused-sgd",
                "--imagenet-pretrained",
            ]
            if (run_dir / "last.pt").is_file():
                command.append("--resume")
            write_state(
                state_path,
                status="running",
                runner_pid=os.getpid(),
                seeds=frozen["seeds"],
                completed_seeds=completed,
                current_seed=seed,
                current_run_dir=relative(run_dir),
            )
            log.write(f"[{timestamp()}] starting seed={seed} command={json.dumps(command)}\n")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            write_state(
                run_dir / "launcher_state.json",
                status="running",
                launcher_pid=os.getpid(),
                training_pid=process.pid,
                seed=seed,
                command=command,
            )
            return_code = process.wait()
            run_state = (
                read_json(run_dir / "run_state.json")
                if (run_dir / "run_state.json").is_file()
                else {}
            )
            if return_code != 0:
                write_state(
                    run_dir / "launcher_state.json",
                    status="failed",
                    return_code=return_code,
                    seed=seed,
                )
                write_state(
                    state_path,
                    status="failed",
                    runner_pid=os.getpid(),
                    seeds=frozen["seeds"],
                    completed_seeds=completed,
                    current_seed=seed,
                    current_run_dir=relative(run_dir),
                    return_code=return_code,
                )
                log.write(f"[{timestamp()}] seed={seed} failed rc={return_code}\n")
                return
            if run_state.get("status") == "paused":
                write_state(
                    run_dir / "launcher_state.json",
                    status="paused",
                    seed=seed,
                    completed_epoch=run_state.get("completed_epoch"),
                )
                write_state(
                    state_path,
                    status="paused",
                    runner_pid=os.getpid(),
                    seeds=frozen["seeds"],
                    completed_seeds=completed,
                    current_seed=seed,
                    current_run_dir=relative(run_dir),
                    completed_epoch=run_state.get("completed_epoch"),
                )
                log.write(f"[{timestamp()}] seed={seed} safely paused\n")
                return
            if run_state.get("status") != "complete":
                raise RuntimeError(
                    f"Seed {seed} exited successfully without a complete run state: {run_state}"
                )
            completed.append(seed)
            write_state(
                run_dir / "launcher_state.json",
                status="complete",
                seed=seed,
            )
            log.write(f"[{timestamp()}] seed={seed} complete\n")
        write_state(
            state_path,
            status="complete",
            runner_pid=os.getpid(),
            seeds=frozen["seeds"],
            completed_seeds=completed,
            current_seed=None,
            current_run_dir=None,
        )
        log.write(f"[{timestamp()}] all seeds complete\n")


if __name__ == "__main__":
    main()
