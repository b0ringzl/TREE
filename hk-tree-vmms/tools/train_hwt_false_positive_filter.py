#!/usr/bin/env python3
"""Train a binary tree/false-positive gate and evaluate it with the species model."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SOURCE_DATA = PROJECT_ROOT / "derived" / "hwt_species_refinement_20260906"
BINARY_DATA = PROJECT_ROOT / "derived" / "hwt_false_positive_filter_20260906"
SPECIES_MODEL = (
    WORKSPACE_ROOT
    / "runs/classify/hk-tree-vmms/derived/training_runs/vmms_species_classifier/"
    / "yolo11s_cls_blocksplit_20260906/weights/best.pt"
)
SPECIES_SUMMARY = PROJECT_ROOT / "derived" / "vmms_species_cls_20260906" / "summary.json"
RUNS = WORKSPACE_ROOT / "runs" / "classify" / "hwt_false_positive_filter"
NEGATIVE = "非树/误检"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA)
    parser.add_argument("--binary-data", type=Path, default=BINARY_DATA)
    parser.add_argument("--species-model", type=Path, default=SPECIES_MODEL)
    parser.add_argument("--species-summary", type=Path, default=SPECIES_SUMMARY)
    parser.add_argument("--model", type=Path, default=WORKSPACE_ROOT / "yolo11s-cls.pt")
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--name", default="yolo11s_hwt_tree_filter_v1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def normalize_species(value: str) -> str:
    folded = value.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return "榕树"
    return "榕树" if value == "榕樹" else value


def materialize(source_data: Path, binary_data: Path) -> dict:
    if binary_data.exists():
        shutil.rmtree(binary_data)
    rows = list(csv.DictReader((source_data / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    counts = Counter()
    targets: dict[tuple[str, str], list[Path]] = {}
    for row in rows:
        split = row["split"]
        binary_class = "1_not_tree" if row["species"] == NEGATIVE else "0_tree"
        source = Path(row["crop"])
        target = binary_data / split / binary_class / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
        targets.setdefault((split, binary_class), []).append(target)
        counts[(split, binary_class)] += 1

    # Equalize the binary training classes. Random image augmentation is applied
    # independently to these linked repeats during training.
    train_target = max(counts[("train", "0_tree")], counts[("train", "1_not_tree")])
    for binary_class in ("0_tree", "1_not_tree"):
        originals = targets[("train", binary_class)]
        index = 0
        while counts[("train", binary_class)] < train_target:
            source = originals[index % len(originals)]
            target = source.with_name(f"{source.stem}__binary_repeat{index:04d}.jpg")
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copy2(source, target)
            counts[("train", binary_class)] += 1
            index += 1
    summary = {
        "policy": "same route-block split as the HWT refinement dataset; balanced binary training only",
        "counts": {f"{split}/{name}": value for (split, name), value in sorted(counts.items())},
    }
    (binary_data / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    cfg = args()
    source_data = cfg.source_data.resolve()
    binary_data = cfg.binary_data.resolve()
    materialize(source_data, binary_data)
    model = YOLO(str(cfg.model.resolve()))
    result = model.train(
        data=str(binary_data), epochs=cfg.epochs, patience=15, batch=64, imgsz=224,
        device=cfg.device, workers=8, project=str(cfg.runs.resolve()), name=cfg.name,
        exist_ok=False, pretrained=True, seed=20260906, deterministic=True,
        optimizer="AdamW", lr0=0.0007, lrf=0.01, weight_decay=0.01,
        warmup_epochs=3, dropout=0.2, auto_augment="randaugment", erasing=0.2,
        hsv_h=0.02, hsv_s=0.45, hsv_v=0.35, fliplr=0.5, verbose=True,
    )
    run_dir = Path(result.save_dir)
    filter_model = YOLO(str(run_dir / "weights" / "best.pt"))
    species_model = YOLO(str(cfg.species_model.resolve()))
    mapping = json.loads(cfg.species_summary.resolve().read_text(encoding="utf-8"))["class_mapping"]
    folder_to_species = {folder: normalize_species(species) for species, folder in mapping.items()}
    manifest = {
        str(Path(row["crop"]).resolve()): row
        for row in csv.DictReader((source_data / "manifest.csv").open(encoding="utf-8-sig", newline=""))
        if row["split"] == "val"
    }
    paths = sorted((binary_data / "val").glob("*/*.jpg"))
    filter_results = filter_model.predict(paths, imgsz=224, batch=128, device=cfg.device, verbose=False)
    species_results = species_model.predict(paths, imgsz=224, batch=128, device=cfg.device, verbose=False)
    observations = []
    for gate, species in zip(filter_results, species_results):
        source_name = Path(gate.path).name
        candidates = [row for path, row in manifest.items() if Path(path).name == source_name]
        row = candidates[0]
        truth = row["species"]
        not_tree_index = next(index for index, name in gate.names.items() if name == "1_not_tree")
        not_tree_probability = float(gate.probs.data[not_tree_index])
        pred_folder = species.names[int(species.probs.top1)]
        pred_species = folder_to_species.get(pred_folder, pred_folder)
        observations.append({
            "path": gate.path, "truth": truth, "species_prediction": pred_species,
            "species_confidence": float(species.probs.top1conf),
            "not_tree_probability": not_tree_probability,
        })
    thresholds = []
    for step in range(101):
        threshold = step / 100
        correct = 0
        rejected_true = rejected_false = 0
        for row in observations:
            reject = row["not_tree_probability"] >= threshold
            if row["truth"] == NEGATIVE:
                correct += int(reject)
                rejected_true += int(reject)
            else:
                correct += int(not reject and row["species_prediction"] == row["truth"])
                rejected_false += int(reject)
        thresholds.append({
            "threshold": threshold, "joint_correct": correct, "total": len(observations),
            "joint_accuracy": correct / len(observations),
            "true_negatives_rejected": rejected_true, "trees_incorrectly_rejected": rejected_false,
        })
    best = max(thresholds, key=lambda row: (row["joint_accuracy"], row["threshold"]))
    baseline_correct = sum(
        row["truth"] != NEGATIVE and row["species_prediction"] == row["truth"]
        for row in observations
    )
    report = {
        "status": "complete", "filter_model": str((run_dir / "weights" / "best.pt").resolve()),
        "species_model": str(cfg.species_model.resolve()), "validation_instances": len(observations),
        "baseline_joint_accuracy": baseline_correct / len(observations),
        "best_two_stage": best,
        "absolute_gain": best["joint_accuracy"] - baseline_correct / len(observations),
        "threshold_sweep": thresholds,
    }
    (run_dir / "two_stage_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "two_stage_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
        writer.writeheader(); writer.writerows(observations)
    print(json.dumps({key: value for key, value in report.items() if key != "threshold_sweep"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
