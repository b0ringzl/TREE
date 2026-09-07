"""Validate a completed B4b TSCMDL fusion run."""

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
from torch import nn
from torch.utils.data import DataLoader


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))
sys.path.insert(0, str(SANDBOX_ROOT / "scripts"))

from tscmdl_whu import classification_metrics  # noqa: E402
from train_b4_tscmdl import (  # noqa: E402
    CachedFeatureDataset,
    make_model,
    run_epoch,
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
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def load_expected_reviews(config: dict[str, object]) -> dict[str, dict[str, str]]:
    review_path = Path(str(config["args"]["quality_review"]))
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    return {
        str(item["sample_key"]): {
            "quality": str(item["quality"]),
            "note": str(item["note"]),
            **({"split": str(item["split"])} if "split" in item else {}),
        }
        for item in payload["reviews"]
    }


def validate_predictions(
    path: Path,
    class_names: list[str],
    expected_sizes: dict[str, int],
    expected_metrics: dict[str, object],
    expected_reviews: dict[str, dict[str, str]],
) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_total = expected_sizes["val"] + expected_sizes["test"]
    if len(rows) != expected_total:
        raise ValueError(f"Prediction row count mismatch: {len(rows)}")
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
    recomputed = {}
    for split in ("val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        truth = []
        predicted = []
        for row in split_rows:
            true_class = int(row["true_class"])
            predicted_class = int(row["predicted_class"])
            probabilities = np.asarray(
                [float(row[field]) for field in probability_fields],
                dtype=np.float64,
            )
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise ValueError(f"Invalid probabilities: {row['sample_key']}")
            if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5):
                raise ValueError(f"Probability sum mismatch: {row['sample_key']}")
            if int(np.argmax(probabilities)) != predicted_class:
                raise ValueError(f"Prediction is not argmax: {row['sample_key']}")
            if int(row["correct"]) != int(true_class == predicted_class):
                raise ValueError(f"Correct flag mismatch: {row['sample_key']}")
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

    row_by_key = {row["sample_key"]: row for row in rows}
    expected_marked_rows = 0
    for sample_key, review in expected_reviews.items():
        row = row_by_key.get(sample_key)
        if row is None:
            continue
        expected_marked_rows += 1
        if "split" in review and row["split"] != review["split"]:
            raise ValueError(f"Review split mismatch: {sample_key}")
        if row["review_quality"] != review["quality"]:
            raise ValueError(f"Review quality mismatch: {sample_key}")
        if row["review_note"] != review["note"]:
            raise ValueError(f"Review note mismatch: {sample_key}")
    marked_rows = [row for row in rows if row["review_quality"]]
    if len(marked_rows) != expected_marked_rows:
        raise ValueError("Quality marker count mismatch")
    return {
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "quality_marked_rows": len(marked_rows),
        "quality_histogram": dict(
            sorted(Counter(row["review_quality"] for row in marked_rows).items())
        ),
        "metrics_recomputed_from_predictions": recomputed,
    }


def independent_checkpoint_inference(
    *,
    config: dict[str, object],
    checkpoint: dict[str, object],
    cache_payload: dict[str, object],
    stored_metrics: dict[str, object],
    predictions_path: Path,
    device: torch.device,
) -> dict[str, object]:
    class_names = [str(value) for value in config["class_names"]]
    architecture = config["fusion_architecture"]
    args = config["args"]
    model = make_model(
        len(class_names),
        float(architecture["dropout"]),
        str(architecture["normalization"]),
        tuple(int(value) for value in architecture["classifier"][:-1]),
        str(architecture["modality"]),
        int(architecture["point_raw"]),
        int(architecture["image_raw"]),
        int(architecture["modal_projection"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with predictions_path.open("r", encoding="utf-8", newline="") as stream:
        stored_rows = list(csv.DictReader(stream))
    stored_by_key = {
        (str(row["split"]), str(row["sample_key"])): row for row in stored_rows
    }
    split_results: dict[str, object] = {}
    max_probability_error = 0.0
    checked_rows = 0
    for split in ("val", "test"):
        dataset = CachedFeatureDataset(cache_payload, split)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(args["eval_batch_size"])),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        metrics, predictions = run_epoch(
            model,
            loader,
            criterion,
            device,
            len(class_names),
            use_amp=bool(args.get("amp", False)) and device.type == "cuda",
            progress_label=f"verify {split}",
            progress_every=1,
            collect_predictions=True,
            show_progress=False,
        )
        expected = stored_metrics[split]
        if metrics["confusion_matrix"] != expected["confusion_matrix"]:
            raise ValueError(f"Independent {split} confusion matrix mismatch")
        for name in (
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ):
            assert_close(
                float(metrics[name]),
                float(expected[name]),
                f"independent {split} {name}",
            )
        for prediction in predictions:
            index = int(prediction["dataset_index"])
            sample_key = dataset.sample_keys[index]
            stored = stored_by_key[(split, sample_key)]
            if int(prediction["predicted_class"]) != int(stored["predicted_class"]):
                raise ValueError(f"Independent predicted class mismatch: {sample_key}")
            for class_index, probability in enumerate(prediction["probabilities"]):
                error = abs(float(probability) - float(stored[f"probability_{class_index}"]))
                max_probability_error = max(max_probability_error, error)
                if error > 1e-5:
                    raise ValueError(
                        f"Independent probability mismatch: {sample_key} class {class_index}"
                    )
            checked_rows += 1
        split_results[split] = metrics
    return {
        "device": str(device),
        "row_count": checked_rows,
        "max_absolute_probability_error": max_probability_error,
        "metrics": split_results,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing B4 artifacts: {missing}")
    temporary_files = sorted(path.name for path in run_dir.glob("*.tmp"))
    if temporary_files:
        raise ValueError(f"Temporary files remain: {temporary_files}")

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    run_type = "b4b_frozen_feature_fusion"
    if config.get("run_type") != run_type or final.get("run_type") != run_type:
        raise ValueError("Run type does not describe B4b fusion")
    if final.get("status") != "complete":
        raise ValueError(f"Run is not complete: {final.get('status')}")
    class_names = [str(value) for value in config["class_names"]]
    split_sizes = {key: int(value) for key, value in config["split_sizes"].items()}
    if len(class_names) < 2 or final["class_names"] != class_names:
        raise ValueError("Class names are inconsistent")
    if set(split_sizes) != {"train", "val", "test"} or any(
        value <= 0 for value in split_sizes.values()
    ):
        raise ValueError(f"Unexpected split sizes: {split_sizes}")
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
        raise ValueError("Best epoch does not match history")
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

    feature_cache = Path(str(config["feature_cache"]["path"]))
    feature_cache_hash = sha256_file(feature_cache)
    if feature_cache_hash != config["feature_cache"]["sha256"]:
        raise ValueError("Feature cache hash differs from run config")
    if feature_cache_hash != final["feature_cache_sha256"]:
        raise ValueError("Feature cache hash differs from final metrics")
    source_hashes = {
        name: sha256_file(Path(path))
        for name, path in config["feature_cache"][
            "source_checkpoint_paths"
        ].items()
    }
    if source_hashes != config["feature_cache"]["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash mismatch")
    cache_payload = torch.load(feature_cache, map_location="cpu", weights_only=False)
    if [str(value) for value in cache_payload["class_names"]] != class_names:
        raise ValueError("Feature cache class order mismatch")
    cache_sizes = {
        split: len(cache_payload["splits"][split]["sample_keys"])
        for split in ("train", "val", "test")
    }
    if cache_sizes != split_sizes:
        raise ValueError(f"Feature cache split sizes mismatch: {cache_sizes}")

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
    for checkpoint in (best_checkpoint, last_checkpoint):
        if checkpoint.get("run_type") != run_type:
            raise ValueError("Checkpoint run type mismatch")
        if list(checkpoint["class_names"]) != class_names:
            raise ValueError("Checkpoint class names mismatch")
        if checkpoint["feature_cache_sha256"] != feature_cache_hash:
            raise ValueError("Checkpoint feature cache hash mismatch")
        state_keys = list(checkpoint["model_state"])
        if not state_keys:
            raise ValueError("Checkpoint model state is empty")
        if any(
            key.startswith(("point_encoder.", "image_encoder."))
            for key in state_keys
        ):
            raise ValueError("Fusion checkpoint unexpectedly embeds a backbone")

    predictions = validate_predictions(
        run_dir / "predictions.csv",
        class_names,
        split_sizes,
        final["metrics"],
        load_expected_reviews(config),
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cpu"
        if args.device == "cpu"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_inference = independent_checkpoint_inference(
        config=config,
        checkpoint=best_checkpoint,
        cache_payload=cache_payload,
        stored_metrics=final["metrics"],
        predictions_path=run_dir / "predictions.csv",
        device=device,
    )
    image_dimensions = {}
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
        "run_type": run_type,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "class_names": class_names,
        "split_sizes": split_sizes,
        "parameter_count": int(config["parameter_count"]),
        "history_epoch_seconds": history_seconds,
        "history_epoch_hours": history_seconds / 3600.0,
        "reported_session_seconds": float(
            final["training_session_elapsed_seconds"]
        ),
        "resumed": bool(final["resumed"]),
        "best_validation_score": final["best_validation_score"],
        "final_metrics": final["metrics"],
        "predictions": predictions,
        "independent_checkpoint_inference": checkpoint_inference,
        "plot_dimensions": image_dimensions,
        "checkpoint_epochs": {
            "best": int(best_checkpoint["epoch"]),
            "last": int(last_checkpoint["epoch"]),
        },
        "feature_cache": {
            "path": str(feature_cache.resolve()),
            "sha256": feature_cache_hash,
            "bytes": feature_cache.stat().st_size,
        },
        "source_checkpoint_sha256": source_hashes,
        "file_sizes_bytes": file_sizes,
        "sha256": {
            name: sha256_file(run_dir / name)
            for name in (
                "best.pt",
                "last.pt",
                "final_metrics.json",
                "predictions.csv",
            )
        },
        "temporary_files": temporary_files,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
