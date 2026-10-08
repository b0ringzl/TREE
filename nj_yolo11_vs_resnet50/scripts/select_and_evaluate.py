"""Select a YOLO11 checkpoint on NJ validation Macro-F1, then evaluate test once."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(dataset_root: Path, split: str) -> list[dict[str, str]]:
    with (dataset_root / "inventory.csv").open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["split"] == split]


def predict(checkpoint: Path, rows: list[dict[str, str]], image_size: int, batch_size: int):
    model = YOLO(str(checkpoint))
    paths = [row["materialized_path"] for row in rows]
    results = model.predict(
        source=paths,
        imgsz=image_size,
        batch=batch_size,
        device=0,
        verbose=False,
    )
    probabilities = np.stack([result.probs.data.cpu().numpy() for result in results])
    true = np.array([int(row["class_index"]) for row in rows], dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    del results, model
    gc.collect()
    torch.cuda.empty_cache()
    return true, predicted, probabilities


def metrics(true: np.ndarray, predicted: np.ndarray, class_names: list[str]) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=list(range(len(class_names))), zero_division=0
    )
    matrix = confusion_matrix(true, predicted, labels=list(range(len(class_names))))
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "sample_count": int(len(true)),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {
                "class_index": index,
                "scientific_name": class_names[index],
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(len(class_names))
        ],
    }


def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
    return int(match.group(1)) + 1 if match else -1


def save_confusion(matrix: list[list[int]], class_names: list[str], path: Path, title: str) -> None:
    array = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(array, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    labels = [name.replace(" ", "\n", 1) for name in class_names]
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = array.max() / 2 if array.size else 0
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            axis.text(
                column,
                row,
                str(array[row, column]),
                ha="center",
                va="center",
                color="white" if array[row, column] > threshold else "black",
            )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "training_complete.json").is_file():
        raise FileNotFoundError(run_dir / "training_complete.json")
    summary = json.loads((dataset_root / "dataset_summary.json").read_text(encoding="utf-8"))
    class_names = list(summary["class_names"])
    val_rows = load_inventory(dataset_root, "val")
    test_rows = load_inventory(dataset_root, "test")
    candidates = sorted((run_dir / "weights").glob("epoch*.pt"), key=checkpoint_epoch)
    if not candidates:
        candidates = [run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"]
    selection_rows: list[dict[str, object]] = []
    started = time.time()
    for checkpoint in candidates:
        true, predicted, _ = predict(checkpoint, val_rows, args.image_size, args.batch_size)
        score = metrics(true, predicted, class_names)
        selection_rows.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": checkpoint_epoch(checkpoint),
                "val_accuracy": score["accuracy"],
                "val_macro_f1": score["macro_f1"],
            }
        )
        print(json.dumps(selection_rows[-1]))
    selected = max(
        selection_rows,
        key=lambda row: (float(row["val_macro_f1"]), float(row["val_accuracy"]), -int(row["epoch"])),
    )
    selected_source = Path(str(selected["checkpoint"]))
    selected_checkpoint = run_dir / "weights" / "selected_by_val_macro_f1.pt"
    shutil.copy2(selected_source, selected_checkpoint)
    with (run_dir / "validation_selection.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    selection_payload = {
        "status": "selection_complete",
        "selection_policy": "validation macro-F1, then accuracy, then earlier epoch",
        "candidate_count": len(selection_rows),
        "selected": selected,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
        "elapsed_seconds": time.time() - started,
        "test_loaded": False,
    }
    (run_dir / "selection_complete.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    evaluation_started = time.time()
    val_true, val_predicted, val_probabilities = predict(
        selected_checkpoint, val_rows, args.image_size, args.batch_size
    )
    test_true, test_predicted, test_probabilities = predict(
        selected_checkpoint, test_rows, args.image_size, args.batch_size
    )
    val_metrics = metrics(val_true, val_predicted, class_names)
    test_metrics = metrics(test_true, test_predicted, class_names)
    save_confusion(
        val_metrics["confusion_matrix"], class_names, run_dir / "confusion_matrix_val.png", "YOLO11s-cls NJ validation"
    )
    save_confusion(
        test_metrics["confusion_matrix"], class_names, run_dir / "confusion_matrix_test.png", "YOLO11s-cls NJ test"
    )
    fields = [
        "sample_key",
        "split",
        "true_class",
        "predicted_class",
        "correct",
        *[f"probability_{index}" for index in range(len(class_names))],
    ]
    prediction_rows: list[dict[str, object]] = []
    for split, rows, truth, predicted, probabilities in (
        ("val", val_rows, val_true, val_predicted, val_probabilities),
        ("test", test_rows, test_true, test_predicted, test_probabilities),
    ):
        for source, true_class, predicted_class, probability in zip(rows, truth, predicted, probabilities):
            row: dict[str, object] = {
                "sample_key": source["sample_key"],
                "split": split,
                "true_class": int(true_class),
                "predicted_class": int(predicted_class),
                "correct": int(true_class == predicted_class),
            }
            row.update({f"probability_{index}": float(value) for index, value in enumerate(probability)})
            prediction_rows.append(row)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    payload = {
        "status": "complete",
        "model": "YOLO11s-cls",
        "class_names": class_names,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": sha256_file(selected_checkpoint),
        "selection": selection_payload,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "test_evaluated_after_training_and_validation_selection": True,
        "evaluation_elapsed_seconds": time.time() - evaluation_started,
        "cuda_peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    (run_dir / "final_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
