"""Recompute critical comparison facts and write a compact acceptance record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


EXPECTED_MANIFEST_SHA256 = "ad3f1be7cf716d50c707148e5870cc7f0dac1eb8c4e903d665653b3b4cffd0e3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    run_dir = parse_args().run_dir.resolve()
    required = [
        run_dir / "dataset" / "dataset_summary.json",
        run_dir / "environment.json",
        run_dir / "training_complete.json",
        run_dir / "selection_complete.json",
        run_dir / "final_metrics.json",
        run_dir / "predictions.csv",
        run_dir / "comparison.json",
        run_dir / "comparison_metrics.png",
        run_dir / "confusion_matrix_test.png",
        run_dir / "weights" / "selected_by_val_macro_f1.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing comparison artifacts: " + ", ".join(missing))
    dataset = read_json(required[0])
    environment = read_json(required[1])
    training = read_json(required[2])
    selection = read_json(required[3])
    metrics = read_json(required[4])
    comparison = read_json(required[6])
    with (run_dir / "results.csv").open("r", encoding="utf-8", newline="") as stream:
        epochs_completed = sum(1 for _ in csv.DictReader(stream))
    with required[5].open("r", encoding="utf-8", newline="") as stream:
        test_rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    true = [int(row["true_class"]) for row in test_rows]
    predicted = [int(row["predicted_class"]) for row in test_rows]
    recomputed_accuracy = accuracy_score(true, predicted)
    recomputed_macro_f1 = f1_score(true, predicted, average="macro", zero_division=0)
    recomputed_matrix = confusion_matrix(true, predicted, labels=[0, 1, 2, 3]).tolist()
    checkpoint = required[-1]
    checks = {
        "source_manifest_hash_matches_frozen_protocol": dataset["source_manifest_sha256"]
        == EXPECTED_MANIFEST_SHA256,
        "dataset_checks_pass": all(dataset["checks"].values()),
        "training_view_excludes_test_directory": not (run_dir / "dataset" / "training_view" / "test").exists(),
        "training_completed": training["status"] == "training_complete",
        "test_not_evaluated_during_training": training["test_loaded_during_training"] is False,
        "selection_completed_before_test": selection["status"] == "selection_complete"
        and selection["test_loaded"] is False,
        "selected_checkpoint_hash_matches": sha256_file(checkpoint)
        == metrics["checkpoint_sha256"],
        "test_count_is_151": len(test_rows) == comparison["test_samples"] == 151,
        "test_accuracy_recomputes": abs(recomputed_accuracy - metrics["metrics"]["test"]["accuracy"]) < 1e-12,
        "test_macro_f1_recomputes": abs(recomputed_macro_f1 - metrics["metrics"]["test"]["macro_f1"]) < 1e-12,
        "test_confusion_matrix_recomputes": recomputed_matrix
        == metrics["metrics"]["test"]["confusion_matrix"],
        "paired_comparison_complete": comparison["status"] == "complete",
        "cuda_used": environment["cuda_available"] is True,
        "ultralytics_version_pinned": environment["ultralytics"] == "8.4.127",
    }
    payload = {
        "status": "passed_with_limitations" if all(checks.values()) else "failed",
        "checks": checks,
        "limitations": comparison["limitations"],
        "observations": {
            "selected_epoch": selection["selected"]["epoch"],
            "epochs_completed": epochs_completed,
            "test_samples": len(test_rows),
            "yolo_accuracy": recomputed_accuracy,
            "resnet_accuracy": comparison["whu_resnet50"]["accuracy"],
            "difference": comparison["paired_accuracy_difference_yolo_minus_resnet"],
        },
    }
    (run_dir / "acceptance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
