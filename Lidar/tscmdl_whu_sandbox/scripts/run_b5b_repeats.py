"""Run the two locked PTv2 repeats required by B5b, one at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]

LOCKED_ARGUMENTS = {
    "epochs": 200,
    "batch_size": 16,
    "eval_batch_size": 16,
    "learning_rate": 0.001,
    "min_learning_rate": 0.00001,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "patience": 40,
    "workers": 0,
    "progress_every": 27,
    "amp": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--knn-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260729, 20260730]
    )
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def assert_baseline_lock(config: dict[str, object]) -> None:
    baseline_args = config["args"]
    for name, expected in LOCKED_ARGUMENTS.items():
        actual = baseline_args[name]
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise ValueError(
                    f"Baseline {name} differs from B5b lock: {actual} != {expected}"
                )
        elif actual != expected:
            raise ValueError(
                f"Baseline {name} differs from B5b lock: {actual} != {expected}"
            )
    if config["model_config"]["pe_multiplier"] is not True:
        raise ValueError("B5b lock requires the PTv2 position multiplier")
    if config["model_config"]["grid_sizes"] != [0.06, 0.12, 0.24, 0.48]:
        raise ValueError("B5b lock requires the B5 grid-size schedule")


def train_command(
    dataset_root: Path,
    cache_path: Path,
    run_dir: Path,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(SANDBOX_ROOT / "scripts" / "train_b5_ptv2.py"),
        "--dataset-root",
        str(dataset_root),
        "--knn-cache",
        str(cache_path),
        "--output-dir",
        str(run_dir),
        "--epochs",
        str(LOCKED_ARGUMENTS["epochs"]),
        "--batch-size",
        str(LOCKED_ARGUMENTS["batch_size"]),
        "--eval-batch-size",
        str(LOCKED_ARGUMENTS["eval_batch_size"]),
        "--learning-rate",
        str(LOCKED_ARGUMENTS["learning_rate"]),
        "--min-learning-rate",
        str(LOCKED_ARGUMENTS["min_learning_rate"]),
        "--weight-decay",
        str(LOCKED_ARGUMENTS["weight_decay"]),
        "--label-smoothing",
        str(LOCKED_ARGUMENTS["label_smoothing"]),
        "--patience",
        str(LOCKED_ARGUMENTS["patience"]),
        "--seed",
        str(seed),
        "--workers",
        "0",
        "--progress-every",
        str(LOCKED_ARGUMENTS["progress_every"]),
    ]


def validate_run(run_dir: Path, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SANDBOX_ROOT / "scripts" / "validate_b5_run.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ],
        cwd=SANDBOX_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Repeat seeds must be non-empty and unique")
    if 20260728 in args.seeds:
        raise ValueError("Seed 20260728 is the completed B5 baseline, not a repeat")

    baseline_run = args.baseline_run.resolve()
    dataset_root = args.dataset_root.resolve()
    cache_path = args.knn_cache.resolve()
    output_root = args.output_root.resolve()
    baseline_config_path = baseline_run / "run_config.json"
    baseline_final_path = baseline_run / "final_metrics.json"
    for path in (
        baseline_config_path,
        baseline_final_path,
        dataset_root / "manifest.json",
        dataset_root / "classes.json",
        cache_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_config = read_json(baseline_config_path)
    baseline_final = read_json(baseline_final_path)
    if baseline_final["status"] != "complete":
        raise ValueError("B5 baseline run is incomplete")
    assert_baseline_lock(baseline_config)

    output_root.mkdir(parents=True, exist_ok=True)
    locked_plan_path = output_root / "locked_plan.json"
    locked_plan = {
        "schema_version": 1,
        "stage": "B5b",
        "status": "locked_before_repeat_test_evaluation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_seed": int(baseline_config["args"]["seed"]),
        "repeat_seeds": args.seeds,
        "baseline_run": str(baseline_run),
        "baseline_run_config_sha256": sha256_file(baseline_config_path),
        "dataset_root": str(dataset_root),
        "manifest_sha256": sha256_file(dataset_root / "manifest.json"),
        "classes_sha256": sha256_file(dataset_root / "classes.json"),
        "knn_cache": str(cache_path),
        "knn_cache_sha256": sha256_file(cache_path),
        "locked_arguments": LOCKED_ARGUMENTS,
        "test_policy": (
            "Each repeat evaluates test only after validation-selected training "
            "finishes; repeat results do not change the locked configuration."
        ),
    }
    if locked_plan_path.exists():
        existing = read_json(locked_plan_path)
        comparable_keys = (
            "stage",
            "status",
            "baseline_seed",
            "repeat_seeds",
            "baseline_run_config_sha256",
            "manifest_sha256",
            "classes_sha256",
            "knn_cache_sha256",
            "locked_arguments",
        )
        for key in comparable_keys:
            if existing[key] != locked_plan[key]:
                raise ValueError(f"Existing B5b lock differs at {key}")
        locked_plan = existing
    else:
        atomic_json(locked_plan_path, locked_plan)

    run_results = []
    for seed in args.seeds:
        run_dir = output_root / f"run_batch16_seed{seed}"
        final_path = run_dir / "final_metrics.json"
        if final_path.is_file() and read_json(final_path).get("status") == "complete":
            print(f"Seed {seed} already complete; validating existing run.", flush=True)
        else:
            if run_dir.exists() and not (run_dir / "last.pt").is_file():
                raise ValueError(f"Incomplete non-resumable run directory: {run_dir}")
            command = train_command(dataset_root, cache_path, run_dir, seed)
            if (run_dir / "last.pt").is_file():
                command.append("--resume")
                print(f"Resuming locked seed {seed}.", flush=True)
            else:
                print(f"Starting locked seed {seed}.", flush=True)
            subprocess.run(command, cwd=SANDBOX_ROOT, check=True)

        validation_path = output_root / f"validation_seed{seed}.json"
        validate_run(run_dir, validation_path)
        final = read_json(final_path)
        validation = read_json(validation_path)
        run_results.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "validation": str(validation_path),
                "validation_status": validation["status"],
                "best_epoch": int(final["best_epoch"]),
                "epochs_completed": int(final["epochs_completed"]),
                "metrics": final["metrics"],
                "training_session_elapsed_seconds": float(
                    final["training_session_elapsed_seconds"]
                ),
            }
        )

    result = {
        "status": "complete",
        "stage": "B5b repeat execution",
        "locked_plan": str(locked_plan_path),
        "baseline_seed": int(baseline_config["args"]["seed"]),
        "repeat_runs": run_results,
    }
    atomic_json(output_root / "repeat_run_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
