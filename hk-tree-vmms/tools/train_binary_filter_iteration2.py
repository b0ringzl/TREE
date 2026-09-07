#!/usr/bin/env python3
"""Fine-tune and calibrate the binary tree / not-tree gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "derived/binary_tree_filter_iteration2_20260906"
DEFAULT_MODEL = (
    WORKSPACE_ROOT / "runs/classify/hwt_false_positive_filter/yolo11s_hwt_tree_filter_v1/weights/best.pt"
)
DEFAULT_FEEDBACK = PROJECT_ROOT / "derived/unsaved_review_feedback_iteration2_20260906"
DEFAULT_RUNS = WORKSPACE_ROOT / "runs/classify/binary_filter_iteration2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--name", default="yolo11s_binary_tree_filter_v2")
    parser.add_argument("--trained-model", type=Path)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def probabilities(
    model_path: Path,
    paths: list[Path],
    device: str,
    truth_by_parent: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not paths:
        return []
    model = YOLO(str(model_path.resolve()))
    not_tree = next(index for index, name in model.names.items() if name == "1_not_tree")
    rows = []
    for result in model.predict(paths, imgsz=224, batch=256, device=device, verbose=False):
        path = Path(result.path).resolve()
        rows.append(
            {
                "path": str(path),
                "truth": (truth_by_parent or {}).get(path.parent.name, path.parent.name),
                "not_tree_probability": float(result.probs.data[not_tree]),
            }
        )
    return rows


def metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positives = [row for row in rows if row["truth"] == "0_tree"]
    negatives = [row for row in rows if row["truth"] == "1_not_tree"]
    tree_recall = (
        sum(row["not_tree_probability"] < threshold for row in positives) / len(positives)
        if positives
        else None
    )
    not_tree_recall = (
        sum(row["not_tree_probability"] >= threshold for row in negatives) / len(negatives)
        if negatives
        else None
    )
    balanced = (
        (tree_recall + not_tree_recall) / 2
        if tree_recall is not None and not_tree_recall is not None
        else None
    )
    return {
        "threshold": threshold,
        "tree_support": len(positives),
        "not_tree_support": len(negatives),
        "tree_recall": tree_recall,
        "not_tree_recall": not_tree_recall,
        "balanced_accuracy": balanced,
    }


def choose_threshold(
    rows: list[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    candidates = [metrics(rows, round(float(value), 3)) for value in np.arange(0.50, 0.981, 0.01)]
    # Also consider lower thresholds: the incumbent was intentionally run at a
    # conservative 0.86, so deployment should be selected by Pareto dominance
    # over its observed tree and not-tree recall rather than by an arbitrary 0.98.
    candidates = [metrics(rows, round(float(value), 3)) for value in np.arange(0.20, 0.981, 0.01)]
    eligible = [
        item
        for item in candidates
        if (item["tree_recall"] or 0) >= (baseline["tree_recall"] or 0)
        and (item["not_tree_recall"] or 0) >= (baseline["not_tree_recall"] or 0)
    ]
    if not eligible:
        eligible = candidates
    return max(
        eligible,
        key=lambda item: (
            item["balanced_accuracy"] or 0,
            item["tree_recall"] or 0,
            item["not_tree_recall"] or 0,
            -item["threshold"],
        ),
    )


def main() -> None:
    cfg = parse_args()
    data = cfg.data.resolve()
    baseline_model = cfg.model.resolve()
    val_paths = image_paths(data / "val")
    feedback_paths = image_paths(cfg.feedback.resolve() / "species_crops") + image_paths(
        cfg.feedback.resolve() / "binary_negative_crops"
    )
    before_val = probabilities(baseline_model, val_paths, cfg.device)
    feedback_truth = {
        "species_crops": "0_tree",
        "binary_negative_crops": "1_not_tree",
    }
    before_feedback = probabilities(
        baseline_model, feedback_paths, cfg.device, feedback_truth
    )
    if cfg.trained_model:
        candidate = cfg.trained_model.resolve()
        run_dir = candidate.parents[1]
    else:
        model = YOLO(str(baseline_model))
        result = model.train(
            data=str(data),
            epochs=cfg.epochs,
            patience=15,
            batch=cfg.batch,
            imgsz=224,
            device=cfg.device,
            workers=cfg.workers,
            project=str(cfg.runs.resolve()),
            name=cfg.name,
            exist_ok=False,
            pretrained=True,
            seed=20260906,
            deterministic=True,
            optimizer="AdamW",
            lr0=0.00025,
            lrf=0.02,
            weight_decay=0.01,
            warmup_epochs=3,
            dropout=0.15,
            auto_augment="randaugment",
            erasing=0.20,
            hsv_h=0.015,
            hsv_s=0.35,
            hsv_v=0.30,
            fliplr=0.5,
            verbose=True,
        )
        run_dir = Path(result.save_dir)
        candidate = run_dir / "weights/best.pt"
    after_val = probabilities(candidate, val_paths, cfg.device)
    after_feedback = probabilities(candidate, feedback_paths, cfg.device, feedback_truth)
    baseline_threshold = metrics(before_val, 0.86)
    selected_threshold = choose_threshold(after_val, baseline_threshold)
    report = {
        "status": "complete",
        "dataset": str(data),
        "baseline_model": str(baseline_model),
        "candidate_model": str(candidate.resolve()),
        "baseline_frozen_validation_at_0_86": baseline_threshold,
        "candidate_frozen_validation": selected_threshold,
        "baseline_feedback_at_0_86": metrics(before_feedback, 0.86),
        "candidate_feedback_at_selected_threshold": metrics(
            after_feedback, selected_threshold["threshold"]
        ),
        "deploy_candidate": (
            selected_threshold["tree_recall"] >= baseline_threshold["tree_recall"]
            and selected_threshold["not_tree_recall"] >= baseline_threshold["not_tree_recall"]
        ),
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
