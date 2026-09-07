"""Validate a finished YOLO11 run and record acceptance checks and warnings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "3f7a79871fe13d3a4724fdf46262f080c535f02e058d9439b01183c7241d843e"


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
    args = parse_args()
    run_dir = args.run_dir.resolve()
    required = {
        "dataset_summary": run_dir / "dataset" / "dataset_summary.json",
        "environment": run_dir / "environment.json",
        "training_complete": run_dir / "training_complete.json",
        "test_metrics": run_dir / "final_test_metrics.json",
        "training_history": run_dir / "results.csv",
        "best_checkpoint": run_dir / "weights" / "best.pt",
        "predictions": run_dir / "test_predictions.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required run artifacts: " + ", ".join(missing))

    dataset = read_json(required["dataset_summary"])
    environment = read_json(required["environment"])
    metrics = read_json(required["test_metrics"])
    with required["training_history"].open("r", encoding="utf-8", newline="") as stream:
        epoch_count = sum(1 for _ in csv.DictReader(stream))

    checkpoint_hash = sha256_file(required["best_checkpoint"])
    finite_metrics = all(
        math.isfinite(float(metrics[key]))
        for key in ("precision", "recall", "map50", "map50_95")
    )
    checks = {
        "source_manifest_hash_matches": dataset["source_manifest_sha256"]
        == EXPECTED_MANIFEST_SHA256,
        "ten_classes_in_every_split": dataset["checks"]["ten_classes_in_every_split"],
        "no_hk_group_leakage": not dataset["checks"]["hk_group_leakage"],
        "no_exact_content_leakage": not dataset["checks"]["exact_content_leakage"],
        "holdout_is_hk_only": dataset["checks"]["holdout_is_hk_only"],
        "cuda_enabled": environment["cuda_available"] is True,
        "ultralytics_version_pinned": environment["ultralytics"] == "8.4.127",
        "training_completed": metrics["test_started_after_training_complete"] is True,
        "checkpoint_hash_matches_test": metrics["checkpoint_sha256"] == checkpoint_hash,
        "test_image_count_matches_dataset": metrics["test_images"]
        == dataset["image_counts"]["test"],
        "test_metrics_are_finite": finite_metrics,
        "predictions_exported": required["predictions"].stat().st_size > 0,
    }
    warnings: list[str] = []
    if float(metrics["recall"]) < 0.5:
        warnings.append("Test recall is below 0.50; the model misses too many target trees.")
    if float(metrics["map50"]) < 0.5:
        warnings.append("Test mAP@0.50 is below 0.50; this checkpoint is not application-ready.")
    per_class = list(metrics["per_class_map50_95"])
    names = {int(key): value for key, value in metrics["class_names"].items()}
    zero_ap = [str(names[index]) for index, value in enumerate(per_class) if float(value) == 0.0]
    if zero_ap:
        warnings.append("Zero test AP for: " + ", ".join(zero_ap))
    small_support = [
        species
        for species, counts in dataset["class_split_box_counts"].items()
        if int(counts["test"]) < 3
    ]
    if small_support:
        warnings.append("Fewer than three test boxes for: " + ", ".join(small_support))

    payload = {
        "status": "passed_with_warnings" if all(checks.values()) else "failed",
        "run_dir": str(run_dir),
        "checks": checks,
        "warnings": warnings,
        "observations": {
            "epochs_completed": epoch_count,
            "test_images": metrics["test_images"],
            "test_boxes": dataset["box_counts"]["test"],
            "predicted_images_at_confidence": metrics.get(
                "predicted_image_count_at_confidence"
            ),
            "prediction_count_at_confidence": metrics["prediction_count_at_confidence"],
            "prediction_confidence": metrics["prediction_confidence"],
            "checkpoint_sha256": checkpoint_hash,
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
