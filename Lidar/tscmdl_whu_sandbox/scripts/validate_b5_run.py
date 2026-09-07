"""Validate a completed PTv2 classification run and emit a report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import uuid
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-class-count", type=int)
    parser.add_argument("--expected-train-size", type=int)
    parser.add_argument("--expected-val-size", type=int)
    parser.add_argument("--expected-test-size", type=int)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
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
    class_names: list[str],
    expected_sizes: dict[str, int],
    expected_metrics: dict[str, object],
) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_total = expected_sizes["val"] + expected_sizes["test"]
    if len(rows) != expected_total:
        raise ValueError(f"Prediction row count mismatch: {len(rows)} != {expected_total}")
    sample_keys = [row["sample_key"] for row in rows]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Duplicate sample keys in predictions")
    split_counts = Counter(row["split"] for row in rows)
    expected_counts = Counter(
        {"val": expected_sizes["val"], "test": expected_sizes["test"]}
    )
    if split_counts != expected_counts:
        raise ValueError(f"Prediction split counts mismatch: {split_counts}")

    num_classes = len(class_names)
    probability_fields = [f"probability_{index}" for index in range(num_classes)]
    recomputed: dict[str, object] = {}
    for split in ("val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        truth: list[int] = []
        predicted: list[int] = []
        for row in split_rows:
            true_class = int(row["true_class"])
            predicted_class = int(row["predicted_class"])
            probabilities = np.asarray(
                [float(row[field]) for field in probability_fields], dtype=np.float64
            )
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
                raise ValueError(f"Invalid probabilities for {row['sample_key']}")
            if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5):
                raise ValueError(f"Probabilities do not sum to one: {row['sample_key']}")
            if int(np.argmax(probabilities)) != predicted_class:
                raise ValueError(f"Predicted class is not argmax: {row['sample_key']}")
            if int(row["correct"]) != int(true_class == predicted_class):
                raise ValueError(f"Correct flag mismatch: {row['sample_key']}")
            if not 0 <= true_class < num_classes:
                raise ValueError(f"True class out of range: {row['sample_key']}")
            if not 0 <= predicted_class < num_classes:
                raise ValueError(f"Predicted class out of range: {row['sample_key']}")
            truth.append(true_class)
            predicted.append(predicted_class)

        metrics = classification_metrics(truth, predicted, num_classes)
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
        "split_counts": dict(sorted(split_counts.items())),
        "metrics_recomputed_from_predictions": recomputed,
    }


def validate_stage_counts(metrics: dict[str, object], split: str) -> None:
    stages = metrics.get("stage_point_counts")
    if not isinstance(stages, list) or len(stages) != 5:
        raise ValueError(f"{split} must report five point-count stages")
    previous_mean = math.inf
    for expected_stage, stage in enumerate(stages):
        if int(stage["stage"]) != expected_stage:
            raise ValueError(f"{split} point-count stage order mismatch")
        minimum = int(stage["minimum"])
        maximum = int(stage["maximum"])
        mean = float(stage["mean"])
        if not 0 < minimum <= mean <= maximum:
            raise ValueError(f"{split} invalid point-count summary at stage {expected_stage}")
        if mean > previous_mean:
            raise ValueError(f"{split} point counts increase after pooling")
        previous_mean = mean
    if int(stages[0]["minimum"]) != 8192 or int(stages[0]["maximum"]) != 8192:
        raise ValueError(f"{split} does not preserve the 8192-point input interface")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing B5 artifacts: {missing}")
    temporary_files = sorted(path.name for path in run_dir.glob("*.tmp"))
    if temporary_files:
        raise ValueError(f"Temporary files remain: {temporary_files}")

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    class_names = [str(value) for value in config["class_names"]]
    split_sizes = {key: int(value) for key, value in config["split_sizes"].items()}
    expected_split_sizes = {
        "train": args.expected_train_size,
        "val": args.expected_val_size,
        "test": args.expected_test_size,
    }
    if any(value is not None for value in expected_split_sizes.values()):
        if any(value is None for value in expected_split_sizes.values()):
            raise ValueError("All three expected split sizes must be provided together")
        if split_sizes != expected_split_sizes:
            raise ValueError(
                f"Unexpected split sizes: {split_sizes} != {expected_split_sizes}"
            )
    if final["status"] != "complete" or not final["test_evaluated_after_training"]:
        raise ValueError("B5 run is incomplete or violates the final-test policy")
    if final["class_names"] != class_names or len(class_names) < 2:
        raise ValueError("Class names are inconsistent")
    if (
        args.expected_class_count is not None
        and len(class_names) != args.expected_class_count
    ):
        raise ValueError(
            f"Unexpected class count: {len(class_names)} != "
            f"{args.expected_class_count}"
        )
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
        key=lambda item: (float(item["val_macro_f1"]), float(item["val_accuracy"])),
    )
    best_epoch = int(final["best_epoch"])
    if int(best_history["epoch"]) != best_epoch:
        raise ValueError("Best epoch does not match the validation history")
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
    if model_config.pop("num_classes") != len(class_names):
        raise ValueError("PTv2 model configuration class count is inconsistent")
    if model_config["pe_multiplier"] is not True or model_config["pe_bias"] is not True:
        raise ValueError("B5 must enable PTv2 position multiplier and bias")
    if model_config["grid_sizes"] != [0.06, 0.12, 0.24, 0.48]:
        raise ValueError("B5 grid-pooling schedule differs from the locked configuration")
    if model_config["encoder_depths"] != [2, 2, 6, 2]:
        raise ValueError("B5 encoder depth differs from the locked PTv2 configuration")
    if config["implementation"]["pointops_extension_used"] is not False:
        raise ValueError("B5 validation expected the native PyTorch backend")
    required_core = {
        "grouped vector attention",
        "grouped linear weight encoding",
        "position encoding multiplier and bias",
        "partition-based grid pooling",
        "global average classification head",
    }
    if set(config["implementation"]["core_modules"]) != required_core:
        raise ValueError("PTv2 core-module declaration is incomplete")

    model = PointTransformerV2Classifier(len(class_names), **model_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["parameter_count"]):
        raise ValueError("Reconstructed PTv2 parameter count mismatch")
    if parameter_count != int(final["parameter_count"]):
        raise ValueError("Final PTv2 parameter count mismatch")

    best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    if int(best_checkpoint["epoch"]) != best_epoch:
        raise ValueError("best.pt epoch mismatch")
    if int(last_checkpoint["epoch"]) != len(history):
        raise ValueError("last.pt epoch mismatch")
    if list(best_checkpoint["class_names"]) != class_names:
        raise ValueError("best.pt class names mismatch")
    if list(last_checkpoint["class_names"]) != class_names:
        raise ValueError("last.pt class names mismatch")
    model.load_state_dict(best_checkpoint["model_state"], strict=True)

    cache = config["knn_cache"]
    cache_path = Path(cache["path"])
    cache_index_path = cache_path / "index.json" if cache_path.is_dir() else cache_path
    if not cache_index_path.is_file():
        raise FileNotFoundError(f"Missing B5 kNN cache: {cache_path}")
    cache_sha256 = sha256_file(cache_index_path)
    if cache_sha256 != cache["sha256"]:
        raise ValueError("B5 kNN cache hash changed after training")
    for checkpoint in (best_checkpoint, last_checkpoint):
        if checkpoint["knn_cache_sha256"] != cache_sha256:
            raise ValueError("Checkpoint kNN cache hash mismatch")
    if final["knn_cache_sha256"] != cache_sha256:
        raise ValueError("Final metrics kNN cache hash mismatch")
    expected_training_tensor_shapes = {
        split: [split_sizes[split], 8192, 8] for split in ("train", "val")
    }
    if cache["tensor_shapes"] != expected_training_tensor_shapes:
        raise ValueError("B5 kNN cache tensor shapes are incorrect")
    if cache_path.is_dir():
        cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
        expected_all_shapes = {
            split: [split_sizes[split], 8192, 8]
            for split in ("train", "val", "test")
        }
        actual_all_shapes = {
            split: list(cache_index["splits"][split]["shape"])
            for split in ("train", "val", "test")
        }
        if actual_all_shapes != expected_all_shapes:
            raise ValueError("Directory kNN cache index split shapes are incorrect")

    predictions = validate_predictions(
        run_dir / "predictions.csv", class_names, split_sizes, final["metrics"]
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
    file_sizes = {name: (run_dir / name).stat().st_size for name in REQUIRED_FILES}
    result = {
        "status": "passed",
        "run_dir": str(run_dir),
        "model": "PTv2 mode-1 classification encoder reimplementation",
        "native_pytorch_backend": True,
        "official_pointops_used": False,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "class_names": class_names,
        "split_sizes": split_sizes,
        "parameter_count": parameter_count,
        "history_epoch_seconds": history_seconds,
        "history_epoch_hours": history_seconds / 3600.0,
        "reported_session_seconds": float(final["training_session_elapsed_seconds"]),
        "resumed": bool(final["resumed"]),
        "best_validation_score": final["best_validation_score"],
        "final_metrics": final["metrics"],
        "predictions": predictions,
        "plot_dimensions": image_dimensions,
        "checkpoint_epochs": {
            "best": int(best_checkpoint["epoch"]),
            "last": int(last_checkpoint["epoch"]),
        },
        "knn_cache": {
            "path": str(cache_index_path),
            "bytes": cache_index_path.stat().st_size,
            "sha256": cache_sha256,
            "tensor_shapes": actual_all_shapes if cache_path.is_dir() else cache["tensor_shapes"],
        },
        "file_sizes_bytes": file_sizes,
        "sha256": {
            name: sha256_file(run_dir / name)
            for name in ("best.pt", "last.pt", "final_metrics.json", "predictions.csv")
        },
        "test_policy": {
            "selection_metric": "validation macro F1 then validation accuracy",
            "test_evaluated_after_training": True,
            "history_contains_test_metrics": any(
                any(key.startswith("test_") for key in item) for item in history
            ),
        },
        "temporary_files": temporary_files,
    }
    if result["test_policy"]["history_contains_test_metrics"]:
        raise ValueError("Training history contains test metrics")
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
