"""Run the frozen-backbone D1 fusion baseline for three matched seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--knn-cache", type=Path, required=True)
    parser.add_argument("--image-suite", type=Path, required=True)
    parser.add_argument("--point-suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-review", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(20260728, 20260729, 20260730)
    )
    parser.add_argument("--cache-batch-size", type=int, default=56)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_state(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    active_seed: int,
    seeds: list[int],
    completed_seeds: list[int],
    message: str,
) -> None:
    atomic_json(
        output_dir / "suite_state.json",
        {
            "updated_at": now(),
            "status": status,
            "phase": phase,
            "active_seed": active_seed,
            "seeds": seeds,
            "completed_seeds": completed_seeds,
            "message": message,
        },
    )


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{now()}] {' '.join(command)}\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=SANDBOX_ROOT.parents[1],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("D1 fusion requires exactly three unique seeds")
    if any(
        value <= 0
        for value in (
            args.cache_batch_size,
            args.epochs,
            args.batch_size,
            args.eval_batch_size,
            args.patience,
        )
    ):
        raise ValueError("Batch sizes, epochs, and patience must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    for name in (
        "dataset_root",
        "knn_cache",
        "image_suite",
        "point_suite",
        "output_dir",
        "quality_review",
    ):
        setattr(args, name, getattr(args, name).resolve())
    protocol_path = args.output_dir / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise ValueError("Fusion protocol must be frozen before launch")
    if [int(value) for value in protocol["seeds"]] != seeds:
        raise ValueError("Runner seeds do not match the frozen protocol")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[int] = []
    launcher_log = args.output_dir / "suite_launcher.log"
    cache_script = SANDBOX_ROOT / "scripts" / "cache_d1_fusion_features.py"
    train_script = SANDBOX_ROOT / "scripts" / "train_b4_tscmdl.py"
    for seed in seeds:
        feature_dir = args.output_dir / "features" / f"seed_{seed}"
        run_dir = args.output_dir / "runs" / f"seed_{seed}"
        feature_path = feature_dir / "features.pt"
        feature_manifest_path = feature_dir / "manifest.json"
        point_checkpoint = args.point_suite / "runs" / f"seed_{seed}" / "best.pt"
        image_checkpoint = args.image_suite / "runs" / f"seed_{seed}" / "best.pt"
        if not point_checkpoint.is_file() or not image_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing matched source checkpoint for seed {seed}"
            )

        feature_ready = False
        if feature_path.is_file() and feature_manifest_path.is_file():
            manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
            feature_ready = (
                manifest.get("status") == "complete"
                and manifest.get("cache_sha256") == sha256_file(feature_path)
                and manifest.get("source_checkpoint_sha256", {}).get("point_best")
                == sha256_file(point_checkpoint)
                and manifest.get("source_checkpoint_sha256", {}).get("image_best")
                == sha256_file(image_checkpoint)
            )
        if not feature_ready:
            if feature_dir.exists() and any(feature_dir.iterdir()):
                raise FileExistsError(
                    f"Incomplete feature directory needs manual inspection: {feature_dir}"
                )
            feature_dir.mkdir(parents=True, exist_ok=True)
            write_state(
                args.output_dir,
                status="running",
                phase="feature_cache",
                active_seed=seed,
                seeds=seeds,
                completed_seeds=completed,
                message=f"Caching matched PTv2/ResNet50 features for seed {seed}",
            )
            run_logged(
                [
                    sys.executable,
                    str(cache_script),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--knn-cache",
                    str(args.knn_cache),
                    "--point-checkpoint",
                    str(point_checkpoint),
                    "--image-checkpoint",
                    str(image_checkpoint),
                    "--output-dir",
                    str(feature_dir),
                    "--batch-size",
                    str(args.cache_batch_size),
                    "--workers",
                    str(args.workers),
                    "--seed",
                    str(seed),
                ],
                launcher_log,
            )

        final_metrics = run_dir / "final_metrics.json"
        if final_metrics.is_file():
            result = json.loads(final_metrics.read_text(encoding="utf-8"))
            if result.get("status") == "complete":
                completed.append(seed)
                continue
        if run_dir.exists() and any(run_dir.iterdir()):
            state_path = run_dir / "run_state.json"
            state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.is_file()
                else {}
            )
            if state.get("status") != "paused":
                raise FileExistsError(
                    f"Incomplete run directory needs manual inspection: {run_dir}"
                )
            resume_flag = ["--resume"]
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            resume_flag = []

        write_state(
            args.output_dir,
            status="running",
            phase="fusion_training",
            active_seed=seed,
            seeds=seeds,
            completed_seeds=completed,
            message=f"Training frozen-backbone fusion head for seed {seed}",
        )
        run_logged(
            [
                sys.executable,
                str(train_script),
                "--feature-cache",
                str(feature_path),
                "--output-dir",
                str(run_dir),
                "--quality-review",
                str(args.quality_review),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--learning-rate",
                "0.01",
                "--min-learning-rate",
                "0.0",
                "--momentum",
                "0.9",
                "--weight-decay",
                "0.0002",
                "--dropout",
                "0.5",
                "--normalization",
                "l2",
                "--classifier-hidden-dims",
                "512",
                "256",
                "--modality",
                "fusion",
                "--patience",
                str(args.patience),
                "--checkpoint-every",
                "1",
                "--seed",
                str(seed),
                "--workers",
                str(args.workers),
                "--progress-every",
                "10",
                "--pause-request",
                str(run_dir / "pause_request.json"),
                "--balanced-sampler",
                *resume_flag,
            ],
            launcher_log,
        )
        if not final_metrics.is_file():
            write_state(
                args.output_dir,
                status="paused",
                phase="fusion_training",
                active_seed=seed,
                seeds=seeds,
                completed_seeds=completed,
                message=f"Seed {seed} paused safely after an epoch checkpoint",
            )
            return
        result = json.loads(final_metrics.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise RuntimeError(f"Seed {seed} did not produce a complete result")
        completed.append(seed)

    write_state(
        args.output_dir,
        status="complete",
        phase="complete",
        active_seed=0,
        seeds=seeds,
        completed_seeds=completed,
        message="All D1 frozen-backbone fusion seeds completed",
    )


if __name__ == "__main__":
    main()
