"""Validate a completed B4c selection and modality-ablation suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import classification_metrics  # noqa: E402


REQUIRED_FILES = (
    "ablation_results.json",
    "best.pt",
    "confusion_matrix_test.png",
    "confusion_matrix_val.png",
    "final_metrics.json",
    "final_summary.json",
    "last.pt",
    "locked_config.json",
    "modality_ablation.png",
    "predictions.csv",
    "regularization_validation.png",
    "run_config.json",
    "selected_fusion_model.json",
    "suite_progress.json",
    "summary.csv",
    "train.log",
    "training_curves.png",
    "training_history.csv",
    "training_history.json",
    "tuning_results.json",
)
MODALITIES = ("fusion", "image", "point")
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


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


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def validate_predictions(
    path: Path,
    ablation: dict[str, object],
    seeds: list[int],
) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_count = len(MODALITIES) * len(seeds) * 2 * 90
    if len(rows) != expected_count:
        raise ValueError(f"Prediction row count mismatch: {len(rows)}")
    compound_keys = [
        (row["modality"], int(row["seed"]), row["split"], row["sample_key"])
        for row in rows
    ]
    if len(compound_keys) != len(set(compound_keys)):
        raise ValueError("Duplicate B4c prediction rows")
    expected_groups = {
        (modality, seed, split)
        for modality in MODALITIES
        for seed in seeds
        for split in ("val", "test")
    }
    group_counts = Counter(
        (row["modality"], int(row["seed"]), row["split"]) for row in rows
    )
    if set(group_counts) != expected_groups or set(group_counts.values()) != {90}:
        raise ValueError(f"Prediction groups are incomplete: {group_counts}")

    stored_runs = {
        (str(run["modality"]), int(run["seed"])): run
        for run in ablation["runs"]
    }
    recomputed = {}
    for modality, seed, split in sorted(expected_groups):
        group = [
            row
            for row in rows
            if row["modality"] == modality
            and int(row["seed"]) == seed
            and row["split"] == split
        ]
        truth = []
        predicted = []
        for row in group:
            probabilities = np.asarray(
                [float(row[f"probability_{index}"]) for index in range(3)]
            )
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise ValueError(f"Invalid probabilities: {row['sample_key']}")
            if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5):
                raise ValueError(f"Probability sum mismatch: {row['sample_key']}")
            true_class = int(row["true_class"])
            predicted_class = int(row["predicted_class"])
            if int(np.argmax(probabilities)) != predicted_class:
                raise ValueError(f"Prediction is not argmax: {row['sample_key']}")
            if int(row["correct"]) != int(true_class == predicted_class):
                raise ValueError(f"Correct flag mismatch: {row['sample_key']}")
            truth.append(true_class)
            predicted.append(predicted_class)
        metrics = classification_metrics(truth, predicted, 3)
        stored = stored_runs[(modality, seed)]["metrics"][split]
        if metrics["confusion_matrix"] != stored["confusion_matrix"]:
            raise ValueError(f"Confusion matrix mismatch: {modality} {seed} {split}")
        for name in METRIC_NAMES:
            assert_close(
                float(metrics[name]),
                float(stored[name]),
                f"{modality} {seed} {split} {name}",
            )
        recomputed[f"{modality}:{seed}:{split}"] = metrics
    return {
        "row_count": len(rows),
        "group_count": len(group_counts),
        "rows_per_group": 90,
        "metrics_recomputed": recomputed,
    }


def aggregate_runs(
    runs: list[dict[str, object]], split: str, metric: str
) -> tuple[float, float]:
    values = np.asarray(
        [float(run["metrics"][split][metric]) for run in runs],
        dtype=np.float64,
    )
    return float(values.mean()), float(values.std(ddof=1))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing B4c artifacts: {missing}")
    temporary_files = sorted(path.name for path in run_dir.rglob("*.tmp"))
    if temporary_files:
        raise ValueError(f"Temporary files remain: {temporary_files}")

    config = read_json(run_dir / "run_config.json")
    tuning = read_json(run_dir / "tuning_results.json")
    locked = read_json(run_dir / "locked_config.json")
    ablation = read_json(run_dir / "ablation_results.json")
    final = read_json(run_dir / "final_metrics.json")
    summary = read_json(run_dir / "final_summary.json")
    selected = read_json(run_dir / "selected_fusion_model.json")
    progress = read_json(run_dir / "suite_progress.json")
    run_type = "b4c_regularization_ablation"
    if config.get("run_type") != run_type or final.get("run_type") != run_type:
        raise ValueError("B4c run type mismatch")
    if any(
        item.get("status") != "complete"
        for item in (ablation, final, summary)
    ):
        raise ValueError("A B4c final artifact is incomplete")
    if tuning.get("status") != "complete":
        raise ValueError("B4c tuning is incomplete")
    if progress.get("phase") != "complete":
        raise ValueError("B4c progress is not complete")

    seeds = [int(value) for value in config["args"]["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("B4c seed set is invalid")
    if tuning.get("test_metrics_present") is not False:
        raise ValueError("Tuning result does not prove test isolation")
    for run in tuning["runs"]:
        if set(run["metrics"]) != {"val"} or run["predictions"]:
            raise ValueError("A tuning run contains test data or predictions")
    if len(tuning["runs"]) != 9:
        raise ValueError("Expected nine validation-only tuning runs")
    if locked.get("status") != "locked_before_test":
        raise ValueError("Configuration was not locked before test")
    if locked.get("test_evaluated_before_lock") is not False:
        raise ValueError("Locked config reports pre-lock test access")
    if sha256_file(run_dir / "tuning_results.json") != locked[
        "tuning_results_sha256"
    ]:
        raise ValueError("Locked tuning-result hash mismatch")
    if ablation.get("test_evaluation_started_after_lock") is not True:
        raise ValueError("Ablation does not confirm post-lock test evaluation")
    if ablation["locked_at"] != locked["locked_at"]:
        raise ValueError("Lock timestamps differ")

    candidates = tuning["candidate_summaries"]
    selected_summary = max(
        candidates,
        key=lambda item: (
            float(item["val_macro_f1_mean"]),
            float(item["val_accuracy_mean"]),
            -int(item["parameter_count"]),
        ),
    )
    if selected_summary["candidate"] != locked["selected_candidate"]["name"]:
        raise ValueError("Locked candidate does not maximize validation selection")
    if locked["selected_candidate"] != ablation["selected_candidate"]:
        raise ValueError("Ablation candidate differs from locked config")
    if len(ablation["runs"]) != 9:
        raise ValueError("Expected nine locked ablation runs")

    for modality in MODALITIES:
        modality_runs = [
            run for run in ablation["runs"] if run["modality"] == modality
        ]
        if sorted(int(run["seed"]) for run in modality_runs) != sorted(seeds):
            raise ValueError(f"Seed coverage mismatch for {modality}")
        stored_summary = next(
            item
            for item in ablation["modality_summaries"]
            if item["modality"] == modality
        )
        for split in ("val", "test"):
            for metric in ("accuracy", "macro_f1"):
                actual_mean, actual_std = aggregate_runs(
                    modality_runs, split, metric
                )
                assert_close(
                    actual_mean,
                    float(stored_summary[f"{split}_{metric}_mean"]),
                    f"{modality} {split} {metric} mean",
                )
                assert_close(
                    actual_std,
                    float(stored_summary[f"{split}_{metric}_std"]),
                    f"{modality} {split} {metric} std",
                )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_paths = sorted(checkpoint_dir.glob("*_best.pt"))
    if len(checkpoint_paths) != 9:
        raise ValueError(f"Expected nine ablation checkpoints: {len(checkpoint_paths)}")
    stored_runs = {
        (str(run["modality"]), int(run["seed"])): run
        for run in ablation["runs"]
    }
    checkpoint_hashes = {}
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        key = (str(checkpoint["modality"]), int(checkpoint["seed"]))
        stored = stored_runs[key]["checkpoint"]
        digest = sha256_file(path)
        if digest != stored["sha256"] or path.stat().st_size != int(stored["bytes"]):
            raise ValueError(f"Checkpoint identity mismatch: {path.name}")
        if checkpoint["candidate"] != locked["selected_candidate"]:
            raise ValueError(f"Checkpoint candidate mismatch: {path.name}")
        checkpoint_hashes[path.name] = digest

    fusion_runs = [
        run for run in ablation["runs"] if run["modality"] == "fusion"
    ]
    expected_selected = max(
        fusion_runs,
        key=lambda run: (
            float(run["metrics"]["val"]["macro_f1"]),
            float(run["metrics"]["val"]["accuracy"]),
        ),
    )
    if int(selected["seed"]) != int(expected_selected["seed"]):
        raise ValueError("Selected fusion seed is not validation-best")
    if int(final["selected_seed"]) != int(selected["seed"]):
        raise ValueError("Final selected seed mismatch")
    if final["metrics"] != expected_selected["metrics"]:
        raise ValueError("Final metrics differ from selected fusion run")

    history = read_json(run_dir / "training_history.json")
    if len(history) != int(final["epochs_completed"]):
        raise ValueError("Selected history length mismatch")
    if [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Selected history epochs are not contiguous")
    best_checkpoint = torch.load(
        run_dir / "best.pt", map_location="cpu", weights_only=False
    )
    last_checkpoint = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=False
    )
    if int(best_checkpoint["epoch"]) != int(final["best_epoch"]):
        raise ValueError("Root best checkpoint epoch mismatch")
    if int(last_checkpoint["epoch"]) != len(history):
        raise ValueError("Root last checkpoint epoch mismatch")

    feature_cache = Path(str(config["feature_cache"]["path"]))
    if sha256_file(feature_cache) != config["feature_cache"]["sha256"]:
        raise ValueError("Feature cache hash mismatch")
    predictions = validate_predictions(
        run_dir / "predictions.csv", ablation, seeds
    )

    image_dimensions = {}
    for name in (
        "training_curves.png",
        "confusion_matrix_val.png",
        "confusion_matrix_test.png",
        "regularization_validation.png",
        "modality_ablation.png",
    ):
        with Image.open(run_dir / name) as image:
            image.verify()
        with Image.open(run_dir / name) as image:
            if image.format != "PNG" or image.width < 500 or image.height < 400:
                raise ValueError(f"Invalid plot artifact: {name}")
            image_dimensions[name] = [image.width, image.height]

    result = {
        "status": "passed",
        "run_dir": str(run_dir),
        "run_type": run_type,
        "selected_candidate": locked["selected_candidate"],
        "selected_fusion_seed": int(selected["seed"]),
        "selected_best_epoch": int(final["best_epoch"]),
        "selected_epochs_completed": int(final["epochs_completed"]),
        "candidate_summaries": candidates,
        "modality_summaries": ablation["modality_summaries"],
        "test_isolation": {
            "tuning_runs": len(tuning["runs"]),
            "tuning_contains_test_metrics": False,
            "lock_status": locked["status"],
            "test_evaluation_started_after_lock": True,
        },
        "predictions": predictions,
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_sha256": checkpoint_hashes,
        "root_checkpoint_sha256": {
            "best.pt": sha256_file(run_dir / "best.pt"),
            "last.pt": sha256_file(run_dir / "last.pt"),
        },
        "image_dimensions": image_dimensions,
        "feature_cache_sha256": config["feature_cache"]["sha256"],
        "temporary_files": temporary_files,
        "suite_elapsed_seconds": float(summary["suite_elapsed_seconds"]),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
