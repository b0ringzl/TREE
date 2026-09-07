#!/usr/bin/env python3
"""Train and evaluate the Ho Man Tin YOLO11 species refinement classifier."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "derived" / "hwt_species_refinement_20260906"
DEFAULT_BASELINE = (
    WORKSPACE_ROOT
    / "runs/classify/hk-tree-vmms/derived/training_runs/vmms_species_classifier/"
    / "yolo11s_cls_blocksplit_20260906/weights/best.pt"
)
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "derived" / "vmms_species_cls_20260906" / "summary.json"
DEFAULT_RUNS = WORKSPACE_ROOT / "runs" / "classify" / "hwt_species_refinement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--refined-model", type=Path)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--name", default="yolo11s_hwt_review_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def canonical_folder(value: str) -> str:
    return value.split("_", 1)[1] if "_" in value else value


def normalize_species(value: str) -> str:
    folded = value.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return "榕树"
    if any(token in value for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return "榕树"
    return "榕树" if value == "榕樹" else value


def evaluate(
    model_path: Path,
    data: Path,
    truth_mapping: dict[str, str],
    prediction_mapping: dict[str, str],
    device: str,
) -> dict:
    paths = sorted((data / "val").glob("*/*.jpg"))
    model = YOLO(str(model_path))
    rows = []
    counts = Counter()
    per_class: dict[str, Counter] = defaultdict(Counter)
    for result in model.predict(paths, imgsz=224, batch=128, device=device, verbose=False):
        truth_folder = Path(result.path).parent.name
        truth = truth_mapping[truth_folder]
        prediction_folder = result.names[int(result.probs.top1)]
        prediction = normalize_species(
            prediction_mapping.get(prediction_folder, prediction_folder)
        )
        # Baseline used separate Ficus heads; merge both during comparison.
        if "Ficus_microcarpa" in prediction_folder or "Ficus_benjamina" in prediction_folder:
            prediction = "榕树"
        correct = prediction == truth
        counts["total"] += 1
        counts["correct"] += int(correct)
        per_class[truth]["support"] += 1
        per_class[truth]["correct"] += int(correct)
        rows.append(
            {
                "path": result.path,
                "truth": truth,
                "prediction": prediction,
                "confidence": float(result.probs.top1conf),
                "correct": correct,
            }
        )
    return {
        "total": counts["total"],
        "correct": counts["correct"],
        "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0,
        "per_class": {
            name: {
                **values,
                "accuracy": values["correct"] / values["support"] if values["support"] else 0,
            }
            for name, values in sorted(per_class.items())
        },
        "rows": rows,
    }


def main() -> None:
    cfg = parse_args()
    data = cfg.data.resolve()
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    species_to_folder = summary["class_mapping"]
    folder_to_species = {folder: species for species, folder in species_to_folder.items()}
    if cfg.dry_run:
        print(json.dumps({"status": "ready", "data": str(data), "classes": len(folder_to_species)}, ensure_ascii=False, indent=2))
        return

    baseline_summary = json.loads(cfg.baseline_summary.resolve().read_text(encoding="utf-8"))
    baseline_folder_to_species = {
        folder: normalize_species(species)
        for species, folder in baseline_summary["class_mapping"].items()
    }
    baseline = evaluate(
        cfg.baseline.resolve(), data, folder_to_species, baseline_folder_to_species, cfg.device
    )
    if cfg.refined_model:
        run_dir = cfg.refined_model.resolve().parents[1]
        best = cfg.refined_model.resolve()
    else:
        model = YOLO(str(cfg.model.resolve()))
        result = model.train(
            data=str(data),
            epochs=cfg.epochs,
            patience=20,
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
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.01,
            warmup_epochs=3,
            dropout=0.20,
            auto_augment="randaugment",
            erasing=0.25,
            hsv_h=0.02,
            hsv_s=0.45,
            hsv_v=0.35,
            fliplr=0.5,
            verbose=True,
        )
        run_dir = Path(result.save_dir)
        best = run_dir / "weights" / "best.pt"
    refined = evaluate(best, data, folder_to_species, folder_to_species, cfg.device)
    report = {
        "status": "complete",
        "dataset": str(data),
        "validation_policy": summary["policy"],
        "validation_blocks": summary["validation_blocks"],
        "baseline_model": str(cfg.baseline.resolve()),
        "refined_model": str(best.resolve()),
        "baseline_accuracy": baseline["accuracy"],
        "refined_accuracy": refined["accuracy"],
        "absolute_gain": refined["accuracy"] - baseline["accuracy"],
        "baseline_per_class": baseline["per_class"],
        "refined_per_class": refined["per_class"],
    }
    (run_dir / "before_after_accuracy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, values in (("baseline", baseline), ("refined", refined)):
        with (run_dir / f"{name}_val_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values["rows"][0]))
            writer.writeheader()
            writer.writerows(values["rows"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
