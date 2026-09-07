#!/usr/bin/env python3
"""Build a route-local visual memory from submitted and autosaved human edits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
COORDINATES = PROJECT_ROOT / "outputs/coordinates/frame_coordinates.csv"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "derived/review_replay_species_iteration2_full_20260906/human_memory/manifest.csv"
)
DEFAULT_MODEL = WORKSPACE_ROOT / "yolo11s-cls.pt"
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "runs/classify/review_replay_species/human_memory_v1"
)
BANYAN = "榕树"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-route-distance-m", type=float, default=250.0)
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
    if any(token in name for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return BANYAN
    return BANYAN if name == "榕樹" else name


def route_distances() -> dict[str, float]:
    with COORDINATES.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["stream_id"]].append(row)
    output = {}
    for stream_id, items in grouped.items():
        total = 0.0
        previous = None
        for row in sorted(items, key=lambda item: int(item["seq_id"])):
            point = (float(row["hk80_easting"]), float(row["hk80_northing"]))
            if previous is not None:
                total += math.dist(previous, point)
            previous = point
            output[f"{stream_id}__{row['frame_id']}"] = total
    return output


def main() -> None:
    cfg = parse_args()
    with cfg.manifest.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    distances = route_distances()
    rows = [row for row in rows if Path(row["crop"]).is_file()]
    for row in rows:
        row["species"] = normalize(row["species"])
        row["route_distance_m"] = distances.get(row["frame_key"])
    model = YOLO(str(cfg.model.resolve()))
    raw = model.embed(
        [row["crop"] for row in rows],
        imgsz=320,
        batch=128,
        device=cfg.device,
        verbose=False,
    )
    embeddings = np.stack([tensor.cpu().numpy() for tensor in raw]).astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    predictions = []
    for index, row in enumerate(rows):
        candidates = [
            other
            for other, value in enumerate(rows)
            if other != index
            and value["frame_key"] != row["frame_key"]
            and value["stream_id"] == row["stream_id"]
            and value["route_distance_m"] is not None
            and row["route_distance_m"] is not None
            and abs(float(value["route_distance_m"]) - float(row["route_distance_m"]))
            <= cfg.maximum_route_distance_m
        ]
        by_species: dict[str, list[int]] = defaultdict(list)
        for other in candidates:
            by_species[rows[other]["species"]].append(other)
        eligible = {
            species: indexes
            for species, indexes in by_species.items()
            if len({rows[other]["frame_key"] for other in indexes}) >= 2
        }
        if not eligible:
            predictions.append(None)
            continue
        scores = sorted(
            (
                (float(np.max(embeddings[indexes] @ embeddings[index])), species)
                for species, indexes in eligible.items()
            ),
            reverse=True,
        )
        predictions.append(
            {
                "species": scores[0][1],
                "similarity": scores[0][0],
                "margin": scores[0][0] - (scores[1][0] if len(scores) > 1 else -1.0),
                "truth": row["species"],
                "frame_key": row["frame_key"],
            }
        )

    rules = []
    trials = []
    for species in sorted({row["species"] for row in rows}, key=str.casefold):
        best = None
        for similarity in np.arange(0.70, 0.991, 0.01):
            for margin in np.arange(0.00, 0.201, 0.02):
                selected = [
                    item
                    for item in predictions
                    if item
                    and item["species"] == species
                    and item["similarity"] >= similarity
                    and item["margin"] >= margin
                ]
                if not selected:
                    continue
                correct = sum(item["truth"] == species for item in selected)
                item = {
                    "species": species,
                    "minimum_similarity": round(float(similarity), 3),
                    "minimum_margin": round(float(margin), 3),
                    "cases": len(selected),
                    "correct": correct,
                    "errors": len(selected) - correct,
                    "precision": correct / len(selected),
                    "query_frames": len({entry["frame_key"] for entry in selected}),
                }
                trials.append(item)
                if item["errors"] == 0 and item["correct"] >= 2 and item["query_frames"] >= 2:
                    if best is None or (
                        item["correct"], item["query_frames"], -item["minimum_similarity"], -item["minimum_margin"]
                    ) > (
                        best["correct"],
                        best["query_frames"],
                        -best["minimum_similarity"],
                        -best["minimum_margin"],
                    ):
                        best = item
        if best:
            rules.append(best)

    output = cfg.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "memory.npz", embeddings=embeddings)
    (output / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    usable = [item for item in predictions if item]
    report = {
        "status": "validated_route_local_memory",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": str(cfg.model.resolve()),
        "memory": str((output / "memory.npz").resolve()),
        "manifest": str((output / "manifest.json").resolve()),
        "maximum_route_distance_m": cfg.maximum_route_distance_m,
        "samples": len(rows),
        "species": dict(Counter(row["species"] for row in rows)),
        "leave_one_frame_out_cases": len(usable),
        "leave_one_frame_out_raw_accuracy": (
            sum(item["species"] == item["truth"] for item in usable) / len(usable)
            if usable
            else None
        ),
        "rules": rules,
        "policy": (
            "same stream, within 250 m, exclude same frame, require at least two distinct memory frames; "
            "deploy per-species thresholds only with zero observed errors on leave-one-frame-out replay"
        ),
    }
    (output / "rules.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "threshold_trials.json").write_text(
        json.dumps(trials, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
