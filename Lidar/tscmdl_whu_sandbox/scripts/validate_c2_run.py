"""Validate the completed 19-class C2 PTv2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    PointTransformerV2Classifier,
    classification_metrics,
)


REQUIRED_FILES = (
    "best.pt",
    "last.pt",
    "run_config.json",
    "training_history.json",
    "training_history.csv",
    "final_metrics.json",
    "predictions.csv",
    "training_curves.png",
    "confusion_matrix_val.png",
    "confusion_matrix_test.png",
    "train.log",
    "run_state.json",
    "launcher_state.json",
    "training_profile.json",
)
EXPECTED_SPLITS = {"train": 13116, "val": 570, "test": 3448}
REQUIRED_DIAGNOSTIC_FILES = (
    "C2_diagnostics.json",
    "C2_per_class_metrics.csv",
    "C2_top_confusions.csv",
    "C2_confusion_matrix_val_readable.png",
    "C2_confusion_matrix_test_readable.png",
    "C2_test_per_class_metrics.png",
    "C2_training_curves_readable.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def validate_predictions(
    path: Path,
    class_names: list[str],
    expected_metrics: dict[str, object],
) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_total = EXPECTED_SPLITS["val"] + EXPECTED_SPLITS["test"]
    if len(rows) != expected_total:
        raise ValueError(f"Prediction row count mismatch: {len(rows)} != {expected_total}")
    sample_keys = [row["sample_key"] for row in rows]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Duplicate sample keys in predictions")
    split_counts = Counter(row["split"] for row in rows)
    if split_counts != Counter({"val": 570, "test": 3448}):
        raise ValueError(f"Prediction split counts mismatch: {split_counts}")

    probability_fields = [
        f"probability_{index}" for index in range(len(class_names))
    ]
    recomputed: dict[str, object] = {}
    maximum_probability_error = 0.0
    for split in ("val", "test"):
        truth: list[int] = []
        predicted: list[int] = []
        for row in (item for item in rows if item["split"] == split):
            true_class = int(row["true_class"])
            predicted_class = int(row["predicted_class"])
            probabilities = np.asarray(
                [float(row[field]) for field in probability_fields],
                dtype=np.float64,
            )
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
                raise ValueError(f"Invalid probabilities for {row['sample_key']}")
            probability_error = abs(float(probabilities.sum()) - 1.0)
            maximum_probability_error = max(
                maximum_probability_error, probability_error
            )
            if probability_error > 1e-5:
                raise ValueError(f"Probabilities do not sum to one: {row['sample_key']}")
            if int(np.argmax(probabilities)) != predicted_class:
                raise ValueError(f"Predicted class is not argmax: {row['sample_key']}")
            if int(row["correct"]) != int(true_class == predicted_class):
                raise ValueError(f"Correct flag mismatch: {row['sample_key']}")
            if not 0 <= true_class < len(class_names):
                raise ValueError(f"True class out of range: {row['sample_key']}")
            if not 0 <= predicted_class < len(class_names):
                raise ValueError(f"Predicted class out of range: {row['sample_key']}")
            truth.append(true_class)
            predicted.append(predicted_class)

        metrics = classification_metrics(truth, predicted, len(class_names))
        stored = expected_metrics[split]
        if metrics["confusion_matrix"] != stored["confusion_matrix"]:
            raise ValueError(f"{split} confusion matrix mismatch")
        for name in (
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ):
            assert_close(float(metrics[name]), float(stored[name]), f"{split} {name}")
        recomputed[split] = metrics
    return {
        "row_count": len(rows),
        "unique_sample_keys": len(set(sample_keys)),
        "split_counts": dict(sorted(split_counts.items())),
        "maximum_probability_sum_error": maximum_probability_error,
        "metrics_recomputed_from_predictions": recomputed,
    }


def validate_stage_counts(metrics: dict[str, object], split: str) -> None:
    stages = metrics.get("stage_point_counts")
    if not isinstance(stages, list) or len(stages) != 5:
        raise ValueError(f"{split} must report five point-count stages")
    previous_mean = math.inf
    for expected_stage, stage in enumerate(stages):
        if int(stage["stage"]) != expected_stage:
            raise ValueError(f"{split} stage order mismatch")
        minimum = int(stage["minimum"])
        maximum = int(stage["maximum"])
        mean = float(stage["mean"])
        if not 0 < minimum <= mean <= maximum or mean > previous_mean:
            raise ValueError(f"{split} invalid stage {expected_stage}")
        previous_mean = mean
    if int(stages[0]["minimum"]) != 8192 or int(stages[0]["maximum"]) != 8192:
        raise ValueError(f"{split} does not preserve the 8192-point input")


def artifact_entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_diagnostics(
    diagnostics_dir: Path,
    run_dir: Path,
    final: dict[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, list[int]]]:
    if not diagnostics_dir.is_dir():
        raise FileNotFoundError(diagnostics_dir)
    missing = [
        name
        for name in REQUIRED_DIAGNOSTIC_FILES
        if not (diagnostics_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing C2 diagnostic artifacts: {missing}")

    diagnostics = read_json(diagnostics_dir / "C2_diagnostics.json")
    if not isinstance(diagnostics, dict) or diagnostics.get("status") != "passed":
        raise ValueError("C2 diagnostics did not pass")
    if Path(str(diagnostics["run_dir"])).resolve() != run_dir:
        raise ValueError("C2 diagnostics refer to a different run directory")
    if int(diagnostics["best_epoch"]) != int(final["best_epoch"]):
        raise ValueError("C2 diagnostics best epoch mismatch")
    for split in ("val", "test"):
        for metric in ("accuracy", "macro_f1", "loss"):
            assert_close(
                float(diagnostics["metrics"][split][metric]),
                float(final["metrics"][split][metric]),
                f"diagnostics {split} {metric}",
            )

    expected_csv_rows = {
        "C2_per_class_metrics.csv": 38,
        "C2_top_confusions.csv": 100,
    }
    for name, minimum_rows in expected_csv_rows.items():
        with (diagnostics_dir / name).open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < minimum_rows:
            raise ValueError(f"Diagnostic CSV is incomplete: {name}")

    image_dimensions: dict[str, list[int]] = {}
    for name in REQUIRED_DIAGNOSTIC_FILES:
        if not name.endswith(".png"):
            continue
        path = diagnostics_dir / name
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG" or image.width < 800 or image.height < 500:
                raise ValueError(f"Invalid readable diagnostic plot: {name}")
            image_dimensions[name] = [image.width, image.height]

    artifacts = {
        name.replace(".", "_"): artifact_entry(diagnostics_dir / name)
        for name in REQUIRED_DIAGNOSTIC_FILES
    }
    return artifacts, image_dimensions


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cache_index = args.cache_index.resolve()
    diagnostics_dir = args.diagnostics_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not cache_index.is_file():
        raise FileNotFoundError(cache_index)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing C2 artifacts: {missing}")
    temporary_files = sorted(path.name for path in run_dir.glob("*.tmp"))
    if temporary_files:
        raise ValueError(f"Temporary files remain: {temporary_files}")

    config = read_json(run_dir / "run_config.json")
    final = read_json(run_dir / "final_metrics.json")
    history = read_json(run_dir / "training_history.json")
    run_state = read_json(run_dir / "run_state.json")
    launcher_state = read_json(run_dir / "launcher_state.json")
    profile = read_json(run_dir / "training_profile.json")
    if not all(
        isinstance(value, dict)
        for value in (config, final, run_state, launcher_state, profile)
    ) or not isinstance(history, list):
        raise ValueError("C2 JSON artifacts have invalid top-level types")

    class_names = [str(value) for value in config["class_names"]]
    split_sizes = {
        key: int(value) for key, value in config["split_sizes"].items()
    }
    if split_sizes != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected C2 split sizes: {split_sizes}")
    if len(class_names) != 19 or final["class_names"] != class_names:
        raise ValueError("C2 must contain the same 19 classes in config and metrics")
    if final["status"] != "complete" or not final["test_evaluated_after_training"]:
        raise ValueError("C2 run is incomplete or violates the final-test policy")
    if run_state.get("status") != "complete":
        raise ValueError("C2 run_state is not complete")
    if launcher_state.get("status") != "complete" or int(
        launcher_state.get("exit_code", -1)
    ) != 0:
        raise ValueError("C2 launcher did not exit successfully")
    if len(history) != int(final["epochs_completed"]):
        raise ValueError("History length does not match epochs_completed")
    if [int(item["epoch"]) for item in history] != list(
        range(1, len(history) + 1)
    ):
        raise ValueError("History epochs are not contiguous")
    if not all(
        math.isfinite(float(value)) for item in history for value in item.values()
    ):
        raise ValueError("History contains a non-finite value")

    best_history = max(
        history,
        key=lambda item: (
            float(item["val_macro_f1"]),
            float(item["val_accuracy"]),
        ),
    )
    best_epoch = int(final["best_epoch"])
    if int(best_history["epoch"]) != best_epoch:
        raise ValueError("Best epoch does not match validation history")
    assert_close(
        float(best_history["val_macro_f1"]),
        float(final["best_validation_score"]["macro_f1"]),
        "best validation macro F1",
    )
    assert_close(
        float(best_history["val_accuracy"]),
        float(final["best_validation_score"]["accuracy"]),
        "best validation accuracy",
    )

    model_config = dict(config["model_config"])
    if int(model_config.pop("num_classes")) != 19:
        raise ValueError("PTv2 model configuration does not target 19 classes")
    if model_config["grid_sizes"] != [0.06, 0.12, 0.24, 0.48]:
        raise ValueError("C2 grid-pooling schedule is not locked")
    if model_config["encoder_depths"] != [2, 2, 6, 2]:
        raise ValueError("C2 encoder depth is not locked")
    if config["implementation"]["pointops_extension_used"] is not False:
        raise ValueError("C2 expected the native PyTorch backend")
    model = PointTransformerV2Classifier(19, **model_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["parameter_count"]):
        raise ValueError("Reconstructed parameter count mismatch")
    if parameter_count != int(final["parameter_count"]):
        raise ValueError("Final parameter count mismatch")

    best_checkpoint = torch.load(
        run_dir / "best.pt", map_location="cpu", weights_only=False
    )
    last_checkpoint = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=False
    )
    if int(best_checkpoint["epoch"]) != best_epoch:
        raise ValueError("best.pt epoch mismatch")
    if int(last_checkpoint["epoch"]) != len(history):
        raise ValueError("last.pt epoch mismatch")
    if int(last_checkpoint["best_epoch"]) != best_epoch:
        raise ValueError("last.pt best epoch mismatch")
    if list(best_checkpoint["class_names"]) != class_names:
        raise ValueError("best.pt class names mismatch")
    if list(last_checkpoint["class_names"]) != class_names:
        raise ValueError("last.pt class names mismatch")
    if len(last_checkpoint.get("history", [])) != len(history):
        raise ValueError("last.pt embedded history mismatch")
    model.load_state_dict(best_checkpoint["model_state"], strict=True)

    cache_sha256 = sha256_file(cache_index)
    if cache_sha256 != config["knn_cache"]["sha256"]:
        raise ValueError("Local C2a cache index differs from C2 config")
    for checkpoint in (best_checkpoint, last_checkpoint):
        if checkpoint["knn_cache_sha256"] != cache_sha256:
            raise ValueError("Checkpoint kNN cache hash mismatch")
    if final["knn_cache_sha256"] != cache_sha256:
        raise ValueError("Final metrics kNN cache hash mismatch")
    if final["test_cache_index_sha256"] != cache_sha256:
        raise ValueError("Final test cache index hash mismatch")
    cache_metadata = read_json(cache_index)
    if not isinstance(cache_metadata, dict):
        raise ValueError("C2a cache index is invalid")
    if cache_metadata["split_sizes"] != EXPECTED_SPLITS:
        raise ValueError("C2a cache split sizes differ from C2")
    if int(cache_metadata["point_count"]) != 8192 or int(
        cache_metadata["neighbours"]
    ) != 8:
        raise ValueError("C2a cache point contract differs from C2")

    predictions = validate_predictions(
        run_dir / "predictions.csv", class_names, final["metrics"]
    )
    for split in ("val", "test"):
        validate_stage_counts(final["metrics"][split], split)

    image_dimensions: dict[str, list[int]] = {}
    for name in (
        "training_curves.png",
        "confusion_matrix_val.png",
        "confusion_matrix_test.png",
    ):
        with Image.open(run_dir / name) as image:
            image.verify()
        with Image.open(run_dir / name) as image:
            if image.format != "PNG" or image.width < 500 or image.height < 400:
                raise ValueError(f"Invalid plot artifact: {name}")
            image_dimensions[name] = [image.width, image.height]

    history_seconds = sum(float(item["epoch_seconds"]) for item in history)
    assert_close(
        float(final["training_elapsed_seconds"]),
        history_seconds,
        "training elapsed seconds",
    )
    training_log = (run_dir / "train.log").read_text(
        encoding="utf-8", errors="replace"
    )
    resume_epochs = [
        int(value) for value in re.findall(r"Resuming from epoch (\d+)", training_log)
    ]
    if "Early stopping after 20 epochs without improvement" not in training_log:
        raise ValueError("C2 log does not contain the expected early-stopping record")
    if int(profile["batch_size"]) != int(config["args"]["batch_size"]):
        raise ValueError("Recovery profile and final run config batch differ")
    if int(profile["eval_batch_size"]) != int(
        config["args"]["eval_batch_size"]
    ):
        raise ValueError("Recovery profile and final run config eval batch differ")

    artifacts = {
        name.replace(".", "_"): artifact_entry(run_dir / name)
        for name in REQUIRED_FILES
    }
    artifacts["cache_index"] = artifact_entry(cache_index)
    diagnostic_artifacts, diagnostic_dimensions = validate_diagnostics(
        diagnostics_dir, run_dir, final
    )
    artifacts.update(diagnostic_artifacts)
    run_files = [path for path in run_dir.iterdir() if path.is_file()]
    result = {
        "status": "passed",
        "stage": "C2b",
        "scope": "completed 19-class C2 PTv2 result closeout",
        "run_dir": str(run_dir),
        "summary": {
            "model": config["model"],
            "class_count": len(class_names),
            "split_sizes": split_sizes,
            "parameter_count": parameter_count,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "resume_epochs": resume_epochs,
            "final_profile": profile,
            "validation_metrics": {
                name: final["metrics"]["val"][name]
                for name in ("loss", "accuracy", "macro_f1", "sample_count")
            },
            "test_metrics": {
                name: final["metrics"]["test"][name]
                for name in ("loss", "accuracy", "macro_f1", "sample_count")
            },
            "result_file_count": len(run_files),
            "result_bytes": sum(path.stat().st_size for path in run_files),
            "history_epoch_seconds": history_seconds,
            "history_epoch_hours": history_seconds / 3600.0,
        },
        "checks": {
            "required_artifacts_present": True,
            "temporary_files_absent": True,
            "run_and_launcher_complete": True,
            "history_contiguous_and_finite": True,
            "best_epoch_matches_history": True,
            "best_and_last_checkpoints_load": True,
            "checkpoint_class_contract_valid": True,
            "model_parameter_contract_valid": True,
            "local_cache_index_hash_matches": True,
            "predictions_complete_and_unique": True,
            "prediction_metrics_recomputed": True,
            "probabilities_valid": True,
            "test_access_locked_until_training_complete": True,
            "early_stopping_recorded": True,
            "recovery_profile_recorded": True,
            "original_plots_are_valid_png": True,
            "readable_diagnostics_complete": True,
        },
        "class_names": class_names,
        "predictions": predictions,
        "plot_dimensions": image_dimensions,
        "diagnostic_plot_dimensions": diagnostic_dimensions,
        "checkpoint_epochs": {
            "best": int(best_checkpoint["epoch"]),
            "last": int(last_checkpoint["epoch"]),
        },
        "sha256": {
            name: sha256_file(run_dir / name)
            for name in (
                "best.pt",
                "last.pt",
                "final_metrics.json",
                "predictions.csv",
            )
        },
        "artifacts": artifacts,
        "warnings": [
            "The run used multiple batch profiles; the best checkpoint was created with batch 48.",
            "The original 19-class annotated confusion matrices are valid but visually crowded.",
            "The benchmark test split is class-imbalanced and is not road-disjoint.",
        ],
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
