#!/usr/bin/env python3
"""Select a YOLO checkpoint by validation group Macro-F1, then test once."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=192)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_for(dataset_root: Path, split: str) -> list[dict[str, str]]:
    with (dataset_root / "inventory.csv").open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["split"] == split]


def predict(checkpoint: Path, rows: list[dict[str, str]], image_size: int, batch_size: int) -> np.ndarray:
    model = YOLO(str(checkpoint))
    chunks = []
    chunk_size = max(batch_size, 512)
    for start in range(0, len(rows), chunk_size):
        batch_rows = rows[start : start + chunk_size]
        results = model.predict(
            source=[row["materialized_path"] for row in batch_rows],
            imgsz=image_size,
            batch=batch_size,
            device=0,
            verbose=False,
        )
        chunks.append(np.stack([result.probs.data.cpu().numpy() for result in results]))
        del results
    probabilities = np.concatenate(chunks, axis=0)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return probabilities


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    probabilities: np.ndarray,
    class_names: list[str],
) -> None:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    fields = [
        "sample_id",
        "group_id",
        "source_path",
        "true_class_index",
        "true_scientific_name",
        "predicted_class_index",
        "predicted_scientific_name",
        "confidence",
        "correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, guess, score in zip(rows, predicted, confidence):
            actual = int(row["class_index"])
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "source_path": row["source_image"],
                    "true_class_index": actual,
                    "true_scientific_name": class_names[actual],
                    "predicted_class_index": int(guess),
                    "predicted_scientific_name": class_names[int(guess)],
                    "confidence": f"{float(score):.8f}",
                    "correct": int(actual == int(guess)),
                }
            )


def metric_values(rows: list[dict[str, str]], probabilities: np.ndarray, class_names: list[str]) -> dict[str, object]:
    truth = np.asarray([int(row["class_index"]) for row in rows])
    predicted = probabilities.argmax(axis=1)
    classes = len(class_names)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        confusion[actual, guess] += 1
    per_class = []
    for index in range(classes):
        tp = int(confusion[index, index]); fp = int(confusion[:, index].sum() - tp); support = int(confusion[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class_index": index, "scientific_name": class_names[index], "support": support, "precision": precision, "recall": recall, "f1": f1})
    top_k = min(5, classes)
    top5 = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    return {
        "accuracy": float(np.mean(truth == predicted)), "top5_accuracy": float(np.mean([actual in guesses for actual, guesses in zip(truth, top5)])),
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])), "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "sample_count": len(rows), "per_class": per_class, "confusion_matrix": confusion.tolist(),
    }


def group_metric_values(rows: list[dict[str, str]], probabilities: np.ndarray, class_names: list[str]) -> dict[str, object]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["group_id"]].append(index)
    group_rows, averages = [], []
    for group_id in sorted(grouped):
        indices = grouped[group_id]
        labels = {rows[index]["class_index"] for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Multiple labels in group {group_id}")
        group_rows.append({"class_index": labels.pop()})
        averages.append(probabilities[indices].mean(axis=0))
    return metric_values(group_rows, np.asarray(averages), class_names)


def epoch_number(path: Path) -> int:
    match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
    return int(match.group(1)) + 1 if match else -1


def main() -> None:
    args = parse_args()
    dataset_root, run_dir = args.dataset_root.resolve(), args.run_dir.resolve()
    if not (run_dir / "training_complete.json").is_file():
        raise FileNotFoundError(run_dir / "training_complete.json")
    summary = json.loads((dataset_root / "dataset_summary.json").read_text(encoding="utf-8"))
    class_names = list(summary["class_names"])
    val_rows = rows_for(dataset_root, "val")
    candidates = sorted((run_dir / "weights").glob("epoch*.pt"), key=epoch_number)
    if not candidates:
        candidates = [run_dir / "weights" / "best.pt"]
    selection_path = run_dir / "validation_selection.csv"
    selected_path = run_dir / "weights" / "selected_by_val_group_macro_f1.pt"
    if selection_path.is_file() and selected_path.is_file():
        numeric_fields = {
            "epoch": int,
            "val_accuracy": float,
            "val_macro_f1": float,
            "val_group_accuracy": float,
            "val_group_macro_f1": float,
        }
        with selection_path.open("r", encoding="utf-8", newline="") as stream:
            selection = list(csv.DictReader(stream))
        for row in selection:
            for field, converter in numeric_fields.items():
                row[field] = converter(row[field])
        print(json.dumps({"status": "reusing_validation_selection", "rows": len(selection)}), flush=True)
    else:
        selection = []
        for checkpoint in candidates:
            probabilities = predict(checkpoint, val_rows, args.image_size, args.batch_size)
            sample = metric_values(val_rows, probabilities, class_names)
            group = group_metric_values(val_rows, probabilities, class_names)
            row = {"checkpoint": str(checkpoint), "epoch": epoch_number(checkpoint), "val_accuracy": sample["accuracy"], "val_macro_f1": sample["macro_f1"], "val_group_accuracy": group["accuracy"], "val_group_macro_f1": group["macro_f1"]}
            selection.append(row)
            print(json.dumps(row), flush=True)
        selected = max(selection, key=lambda row: (row["val_group_macro_f1"], row["val_group_accuracy"], -row["epoch"]))
        shutil.copy2(Path(selected["checkpoint"]), selected_path)
        with selection_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selection[0]))
            writer.writeheader(); writer.writerows(selection)
    selected = max(selection, key=lambda row: (row["val_group_macro_f1"], row["val_group_accuracy"], -row["epoch"]))
    val_probabilities = predict(selected_path, val_rows, args.image_size, args.batch_size)
    test_rows = rows_for(dataset_root, "test")
    test_probabilities = predict(selected_path, test_rows, args.image_size, args.batch_size)
    write_predictions(run_dir / "val_predictions.csv", val_rows, val_probabilities, class_names)
    write_predictions(run_dir / "test_predictions.csv", test_rows, test_probabilities, class_names)
    final = {
        "status": "complete", "model": "YOLO11s-cls", "class_names": class_names,
        "checkpoint": str(selected_path), "checkpoint_sha256": sha256_file(selected_path), "selection": selected,
        "split_sizes": summary["split_sizes"],
        "metrics": {
            "val_sample": metric_values(val_rows, val_probabilities, class_names), "val_group": group_metric_values(val_rows, val_probabilities, class_names),
            "test_sample": metric_values(test_rows, test_probabilities, class_names), "test_group": group_metric_values(test_rows, test_probabilities, class_names),
        },
        "test_evaluated_after_validation_selection": True,
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_epoch": selected["epoch"],
                "test_sample": {
                    key: final["metrics"]["test_sample"][key]
                    for key in ("accuracy", "top5_accuracy", "balanced_accuracy", "macro_f1", "sample_count")
                },
                "test_group": {
                    key: final["metrics"]["test_group"][key]
                    for key in ("accuracy", "top5_accuracy", "balanced_accuracy", "macro_f1", "sample_count")
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
