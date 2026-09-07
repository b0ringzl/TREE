#!/usr/bin/env python3
"""Train the species-only replay classifier and compare it with the old model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "derived" / "review_replay_species_20260906"
DEFAULT_BASELINE = (
    WORKSPACE_ROOT
    / "runs/classify/hk-tree-vmms/derived/training_runs/vmms_species_classifier/"
    / "yolo11s_cls_blocksplit_20260906/weights/best.pt"
)
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "derived" / "vmms_species_cls_20260906" / "summary.json"
DEFAULT_RUNS = WORKSPACE_ROOT / "runs" / "classify" / "review_replay_species"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=WORKSPACE_ROOT / "yolo11s-cls.pt")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--name", default="yolo11s_review_replay_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trained-model", type=Path)
    return parser.parse_args()


def normalize_species(value: str) -> str:
    folded = value.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return "榕树"
    if any(token in value for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return "榕树"
    return "榕树" if value == "榕樹" else value


def evaluate(
    model_path: Path,
    paths: list[Path],
    truth_by_path: dict[str, str],
    folder_to_species: dict[str, str],
    device: str,
) -> dict:
    model = YOLO(str(model_path))
    rows = []
    per_class: dict[str, Counter] = defaultdict(Counter)
    top1 = top5 = 0
    for result in model.predict(paths, imgsz=320, batch=128, device=device, verbose=False):
        truth = truth_by_path[str(Path(result.path).resolve())]
        order = [int(index) for index in result.probs.top5]
        predictions = [normalize_species(folder_to_species.get(result.names[index], result.names[index])) for index in order]
        correct = predictions[0] == truth
        in_top5 = truth in predictions
        top1 += int(correct)
        top5 += int(in_top5)
        per_class[truth]["support"] += 1
        per_class[truth]["correct"] += int(correct)
        rows.append(
            {
                "path": result.path,
                "truth": truth,
                "prediction": predictions[0],
                "confidence": float(result.probs.top1conf),
                "correct": correct,
                "top5_correct": in_top5,
            }
        )
    class_metrics = {
        name: {
            "support": values["support"],
            "correct": values["correct"],
            "accuracy": values["correct"] / values["support"],
        }
        for name, values in sorted(per_class.items())
    }
    supported = [value["accuracy"] for value in class_metrics.values() if value["support"]]
    return {
        "total": len(rows),
        "top1": top1 / len(rows) if rows else 0.0,
        "top5": top5 / len(rows) if rows else 0.0,
        "macro_recall": sum(supported) / len(supported) if supported else 0.0,
        "per_class": class_metrics,
        "rows": rows,
    }


def main() -> None:
    cfg = parse_args()
    data = cfg.data.resolve()
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    folder_to_species = {folder: species for species, folder in summary["class_mapping"].items()}
    manifest = list(csv.DictReader((data / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    val_rows = [row for row in manifest if row["split"] == "val"]
    paths = [Path(row["crop"]).resolve() for row in val_rows]
    truth_by_path = {str(Path(row["crop"]).resolve()): row["species"] for row in val_rows}

    baseline_summary = json.loads(cfg.baseline_summary.resolve().read_text(encoding="utf-8"))
    baseline_mapping = {
        folder: normalize_species(species)
        for species, folder in baseline_summary["class_mapping"].items()
    }
    baseline = evaluate(cfg.baseline.resolve(), paths, truth_by_path, baseline_mapping, cfg.device)

    if cfg.trained_model:
        best = cfg.trained_model.resolve()
        run_dir = best.parents[1]
    else:
        model = YOLO(str(cfg.model.resolve()))
        result = model.train(
            data=str(data),
            epochs=cfg.epochs,
            patience=20,
            batch=cfg.batch,
            imgsz=320,
            device=cfg.device,
            workers=cfg.workers,
            project=str(cfg.runs.resolve()),
            name=cfg.name,
            exist_ok=False,
            pretrained=True,
            seed=20260906,
            deterministic=True,
            optimizer="AdamW",
            lr0=0.0008,
            lrf=0.02,
            weight_decay=0.01,
            warmup_epochs=4,
            dropout=0.15,
            auto_augment="randaugment",
            erasing=0.15,
            hsv_h=0.015,
            hsv_s=0.35,
            hsv_v=0.30,
            fliplr=0.5,
            verbose=True,
        )
        run_dir = Path(result.save_dir)
        best = run_dir / "weights" / "best.pt"

    refined = evaluate(best, paths, truth_by_path, folder_to_species, cfg.device)
    hwt_paths = {
        str(Path(row["crop"]).resolve())
        for row in val_rows
        if row["source_kind"].startswith("human_review_correction")
    }

    def subset(metric: dict, selected: set[str]) -> dict:
        rows = [row for row in metric["rows"] if str(Path(row["path"]).resolve()) in selected]
        return {
            "total": len(rows),
            "top1": sum(row["correct"] for row in rows) / len(rows) if rows else None,
            "top5": sum(row["top5_correct"] for row in rows) / len(rows) if rows else None,
        }

    report = {
        "status": "complete",
        "architecture": "tree detector -> binary false-positive gate -> species-only replay classifier",
        "data": str(data),
        "model": str(best.resolve()),
        "validation_policy": summary["policy"],
        "overall_validation": {
            "baseline_top1": baseline["top1"],
            "refined_top1": refined["top1"],
            "absolute_gain": refined["top1"] - baseline["top1"],
            "baseline_macro_recall": baseline["macro_recall"],
            "refined_macro_recall": refined["macro_recall"],
        },
        "held_out_human_corrections": {
            "baseline": subset(baseline, hwt_paths),
            "refined": subset(refined, hwt_paths),
        },
        "baseline_per_class": baseline["per_class"],
        "refined_per_class": refined["per_class"],
    }
    (run_dir / "review_replay_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, metric in (("baseline", baseline), ("refined", refined)):
        with (run_dir / f"{name}_validation_predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric["rows"][0]))
            writer.writeheader()
            writer.writerows(metric["rows"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
