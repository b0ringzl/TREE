#!/usr/bin/env python3
"""Select zero-regression specialist overrides on the frozen species validation set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "derived/review_replay_species_iteration2_full_20260906"
DEFAULT_ROUTER = (
    WORKSPACE_ROOT / "runs/classify/review_replay_species/species_fusion_adapter_v1/router_rules.json"
)
DEFAULT_V3 = (
    WORKSPACE_ROOT
    / "runs/classify/review_replay_species/yolo11s_review_replay_v3_full_feedback/weights/best.pt"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "runs/classify/review_replay_species/species_fusion_adapter_v2"
)
BANYAN = "榕树"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--v3", type=Path, default=DEFAULT_V3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def normalize(value: object) -> str:
    name = str(value or "").strip()
    folded = name.casefold()
    if folded.startswith("albizia lebbeck"):
        return "Albizia lebbeck"
    if folded.startswith("aleurites moluccana"):
        return "Aleurites moluccana 石栗"
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return BANYAN
    return BANYAN if name == "榕樹" else name


def infer(
    model_path: Path,
    paths: list[Path],
    mapping: dict[str, str],
    device: str,
) -> list[dict[str, Any]]:
    model = YOLO(str(model_path.resolve()))
    rows = []
    for result in model.predict(paths, imgsz=320, batch=256, device=device, verbose=False):
        index = int(result.probs.top1)
        folder = result.names[index]
        rows.append(
            {
                "prediction": normalize(mapping.get(folder, folder)),
                "confidence": float(result.probs.top1conf),
            }
        )
    return rows


def accuracy(predictions: list[str], truth: list[str]) -> float:
    return sum(left == right for left, right in zip(predictions, truth)) / len(truth)


def main() -> None:
    cfg = parse_args()
    data = cfg.data.resolve()
    router = json.loads(cfg.router.resolve().read_text(encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    folder_to_species = {
        folder: normalize(species) for species, folder in summary["class_mapping"].items()
    }
    with (data / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        val = [row for row in csv.DictReader(handle) if row["split"] == "val"]
    paths = [Path(row["crop"]).resolve() for row in val]
    truth = [normalize(row["species"]) for row in val]
    old = infer(Path(router["old_model"]), paths, router["old_mapping"], cfg.device)
    v1 = infer(Path(router["new_model"]), paths, router["new_mapping"], cfg.device)
    v3 = infer(cfg.v3.resolve(), paths, folder_to_species, cfg.device)

    current = []
    legacy = router.get("specialist_override") or {}
    for old_row, v1_row in zip(old, v1):
        chosen = old_row["prediction"]
        if (
            v1_row["prediction"] == legacy.get("species")
            and v1_row["confidence"] >= float(legacy.get("minimum_confidence", 1.1))
        ):
            chosen = v1_row["prediction"]
        current.append(chosen)

    thresholds = [round(value / 100, 2) for value in range(50, 100)]
    selected = []
    candidates = []
    for species in sorted(set(truth), key=str.casefold):
        best = None
        for threshold in thresholds:
            indexes = [
                index
                for index, row in enumerate(v3)
                if row["prediction"] == species
                and row["confidence"] >= threshold
                and current[index] != species
            ]
            if not indexes:
                continue
            correct = sum(truth[index] == species for index in indexes)
            errors = len(indexes) - correct
            current_correct = sum(current[index] == truth[index] for index in indexes)
            item = {
                "species": species,
                "minimum_confidence": threshold,
                "override_cases": len(indexes),
                "override_correct": correct,
                "override_errors": errors,
                "net_gain": correct - current_correct,
                "precision": correct / len(indexes),
            }
            candidates.append(item)
            if errors == 0 and correct >= 2:
                if best is None or (item["net_gain"], item["override_cases"], -threshold) > (
                    best["net_gain"], best["override_cases"], -best["minimum_confidence"]
                ):
                    best = item
        if best:
            selected.append({**best, "model_key": "iteration3"})

    routed = list(current)
    for index, row in enumerate(v3):
        for rule in selected:
            if (
                row["prediction"] == rule["species"]
                and row["confidence"] >= rule["minimum_confidence"]
                and routed[index] != rule["species"]
            ):
                routed[index] = row["prediction"]
                break
    per_class = {}
    for species in sorted(set(truth), key=str.casefold):
        indexes = [index for index, value in enumerate(truth) if value == species]
        per_class[species] = {
            "support": len(indexes),
            "current_correct": sum(current[index] == species for index in indexes),
            "routed_correct": sum(routed[index] == species for index in indexes),
        }

    output = cfg.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        **router,
        "status": "validated_multi_specialist_router",
        "policy": (
            "Retain the original model and validated V1 Araucaria override; add only V3 top-1 "
            "overrides with zero errors and at least two corrections on the frozen validation set."
        ),
        "iteration3_model": str(cfg.v3.resolve()),
        "iteration3_mapping": folder_to_species,
        "specialist_overrides": [
            {
                **rule,
                "validation_policy": "zero observed override errors; >=2 corrected cases",
            }
            for rule in selected
        ],
        "validation": {
            "total": len(truth),
            "old_accuracy": accuracy([row["prediction"] for row in old], truth),
            "current_router_accuracy": accuracy(current, truth),
            "iteration3_direct_accuracy": accuracy([row["prediction"] for row in v3], truth),
            "iteration3_routed_accuracy": accuracy(routed, truth),
            "net_correct_gain": sum(left == right for left, right in zip(routed, truth))
            - sum(left == right for left, right in zip(current, truth)),
            "per_class": per_class,
        },
    }
    (output / "router_rules.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "candidate_thresholds.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
