#!/usr/bin/env python3
"""Learn a small, block-separated adapter over old and review-replay YOLO logits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA = PROJECT_ROOT / "derived" / "review_replay_species_20260906"
OLD_MODEL = (
    WORKSPACE_ROOT
    / "runs/classify/hk-tree-vmms/derived/training_runs/vmms_species_classifier/"
    / "yolo11s_cls_blocksplit_20260906/weights/best.pt"
)
OLD_SUMMARY = PROJECT_ROOT / "derived" / "vmms_species_cls_20260906" / "summary.json"
NEW_MODEL = WORKSPACE_ROOT / "runs/classify/review_replay_species/yolo11s_review_replay_v1/weights/best.pt"
OUTPUT = WORKSPACE_ROOT / "runs" / "classify" / "review_replay_species" / "species_fusion_adapter_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--old-model", type=Path, default=OLD_MODEL)
    parser.add_argument("--old-summary", type=Path, default=OLD_SUMMARY)
    parser.add_argument("--new-model", type=Path, default=NEW_MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def normalize_species(value: str) -> str:
    folded = value.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return "榕树"
    if any(token in value for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return "榕树"
    return "榕树" if value == "榕樹" else value


def stable_fold(value: str) -> int:
    return int(hashlib.sha256(f"20260906:{value}".encode()).hexdigest()[:8], 16) % 5


def mapped_probabilities(
    model_path: Path,
    paths: list[Path],
    model_folder_to_species: dict[str, str],
    species: list[str],
    device: str,
) -> np.ndarray:
    model = YOLO(str(model_path))
    output = np.zeros((len(paths), len(species)), dtype=np.float32)
    target_index = {name: index for index, name in enumerate(species)}
    cursor = 0
    for start in range(0, len(paths), 256):
        batch = paths[start : start + 256]
        results = model.predict(batch, imgsz=320, batch=128, device=device, verbose=False)
        for result in results:
            for index, value in enumerate(result.probs.data.cpu().numpy()):
                name = normalize_species(model_folder_to_species.get(result.names[index], result.names[index]))
                if name in target_index:
                    output[cursor, target_index[name]] += float(value)
            cursor += 1
    return output


def features(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    eps = 1e-6
    old_sorted = np.sort(old, axis=1)
    new_sorted = np.sort(new, axis=1)
    scalars = np.column_stack(
        (
            old_sorted[:, -1],
            old_sorted[:, -1] - old_sorted[:, -2],
            new_sorted[:, -1],
            new_sorted[:, -1] - new_sorted[:, -2],
        )
    )
    return np.column_stack((np.log(np.clip(old, eps, 1.0)), np.log(np.clip(new, eps, 1.0)), scalars))


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    per_class: dict[str, Counter] = defaultdict(Counter)
    for expected, actual in zip(truth, prediction):
        per_class[str(expected)]["support"] += 1
        per_class[str(expected)]["correct"] += int(expected == actual)
    return {
        "total": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "per_class": {
            name: {
                "support": values["support"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["support"],
            }
            for name, values in sorted(per_class.items())
        },
    }


def main() -> None:
    cfg = parse_args()
    data = cfg.data.resolve()
    output = cfg.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    new_mapping = {folder: species for species, folder in summary["class_mapping"].items()}
    old_summary = json.loads(cfg.old_summary.resolve().read_text(encoding="utf-8"))
    old_mapping = {
        folder: normalize_species(species)
        for species, folder in old_summary["class_mapping"].items()
    }
    species = sorted(summary["class_mapping"], key=str.casefold)
    rows = list(csv.DictReader((data / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    rows = [row for row in rows if not row["source_kind"].endswith("balanced_repeat")]
    paths = [Path(row["crop"]).resolve() for row in rows]
    truth = np.asarray([row["species"] for row in rows], dtype=object)
    old_probs = mapped_probabilities(cfg.old_model.resolve(), paths, old_mapping, species, cfg.device)
    new_probs = mapped_probabilities(cfg.new_model.resolve(), paths, new_mapping, species, cfg.device)
    matrix = features(old_probs, new_probs)
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    val_mask = ~train_mask
    inner_cal = np.asarray(
        [row["split"] == "train" and stable_fold(row["route_block_id"]) == 0 for row in rows]
    )
    inner_train = train_mask & ~inner_cal

    candidates = []
    for class_weight in (None, "balanced"):
        for c_value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            adapter = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=3000,
                    solver="lbfgs",
                ),
            )
            adapter.fit(matrix[inner_train], truth[inner_train])
            predicted = adapter.predict(matrix[inner_cal])
            score = accuracy_score(truth[inner_cal], predicted) + 0.20 * balanced_accuracy_score(
                truth[inner_cal], predicted
            )
            candidates.append(
                {
                    "C": c_value,
                    "class_weight": class_weight,
                    "score": float(score),
                    "accuracy": float(accuracy_score(truth[inner_cal], predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(truth[inner_cal], predicted)),
                }
            )
    selected = max(candidates, key=lambda row: (row["score"], -row["C"]))
    adapter = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=selected["C"],
            class_weight=selected["class_weight"],
            max_iter=3000,
            solver="lbfgs",
        ),
    )
    adapter.fit(matrix[train_mask], truth[train_mask])

    old_prediction = np.asarray([species[index] for index in old_probs.argmax(axis=1)], dtype=object)
    new_prediction = np.asarray([species[index] for index in new_probs.argmax(axis=1)], dtype=object)
    fused_prediction = adapter.predict(matrix)
    hwt_val = val_mask & np.asarray(
        [row["source_kind"].startswith("human_review_correction") for row in rows]
    )
    report = {
        "status": "complete",
        "species": species,
        "feature_order": ["old_log_probs", "review_replay_log_probs", "old_conf", "old_margin", "new_conf", "new_margin"],
        "selected_inner_calibration": selected,
        "inner_candidates": candidates,
        "validation": {
            "old": metrics(truth[val_mask], old_prediction[val_mask]),
            "review_replay": metrics(truth[val_mask], new_prediction[val_mask]),
            "fusion_adapter": metrics(truth[val_mask], fused_prediction[val_mask]),
        },
        "held_out_human_corrections": {
            "old": metrics(truth[hwt_val], old_prediction[hwt_val]),
            "review_replay": metrics(truth[hwt_val], new_prediction[hwt_val]),
            "fusion_adapter": metrics(truth[hwt_val], fused_prediction[hwt_val]),
        },
        "models": {
            "old": str(cfg.old_model.resolve()),
            "review_replay": str(cfg.new_model.resolve()),
            "adapter": str((output / "adapter.joblib").resolve()),
        },
    }
    joblib.dump(
        {
            "adapter": adapter,
            "species": species,
            "old_mapping": old_mapping,
            "new_mapping": new_mapping,
            "old_model": str(cfg.old_model.resolve()),
            "new_model": str(cfg.new_model.resolve()),
            "feature_version": "old_new_log_probs_conf_margin_v1",
        },
        output / "adapter.joblib",
    )
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
