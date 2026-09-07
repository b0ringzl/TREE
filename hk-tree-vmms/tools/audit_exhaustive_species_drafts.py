#!/usr/bin/env python3
"""Audit automatic VMMS drafts against every available prior human record."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_NAMES = {"Unknown / 待定", "Unknown 未知", ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, required=True)
    return parser.parse_args()


def canonical(value: str) -> str:
    value = re.sub(r"^\d+_", "", value).replace("_", " ")
    value = value.replace("(syn.", " (").replace("（", "(")
    words = value.split("(", 1)[0].strip().split()
    return " ".join(words[:2]).casefold()


def frame_identity(record: dict[str, Any], path: Path) -> tuple[str, str, str] | None:
    stream = record.get("stream_id") or record.get("frame", {}).get("stream_id")
    frame_id = str(record.get("frame_id") or record.get("frame", {}).get("frame_id") or "")
    text = path.as_posix().casefold()
    if "hewentian_frame_labeler" in text:
        route, stream = "hewentian", stream or "hewentian_pano"
    elif "jianshazui_frame_labeler" in text:
        route = "jianshazui"
    elif "stubbs_road_frame_labeler" in text:
        route = "stubbs_road"
    else:
        return None
    if not stream or not frame_id:
        return None
    return route, str(stream), frame_id


def load_human_records() -> dict[tuple[str, str, str], tuple[dict[str, Any], Path]]:
    selected: dict[tuple[str, str, str], tuple[dict[str, Any], Path, int, float]] = {}
    for root in sorted((PROJECT_ROOT / "derived").glob("*_frame_labeler")):
        candidates = list((root / "annotations" / "records").rglob("*.json"))
        candidates.extend((root / "training_exports").glob("*/records/**/*.json"))
        for path in candidates:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            identity = frame_identity(record, path)
            if identity is None or "labels" not in record:
                continue
            live = int("/annotations/records/" in path.as_posix())
            rank = (live, path.stat().st_mtime)
            current = selected.get(identity)
            if current is None or rank > (current[2], current[3]):
                selected[identity] = (record, path, *rank)
    return {key: (value[0], value[1]) for key, value in selected.items()}


def raster(points: list[list[float]], size: tuple[int, int] = (512, 256)) -> np.ndarray:
    image = Image.new("1", size, 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(round(x * size[0]), round(y * size[1])) for x, y in points], fill=1)
    return np.asarray(image, dtype=bool)


def iou(first: list[list[float]], second: list[list[float]]) -> float:
    a, b = raster(first), raster(second)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def main() -> None:
    args = parse_args()
    draft_dir = args.draft_dir.resolve()
    records_dir = draft_dir / "records"
    if not records_dir.is_dir():
        raise FileNotFoundError(records_dir)
    human_records = load_human_records()
    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for (route, stream, frame_id), (human, source_path) in sorted(human_records.items()):
        draft_path = records_dir / f"{route}__{stream}__{frame_id}.json"
        if not draft_path.is_file():
            counters["human_frames_outside_draft_sample"] += 1
            continue
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        human_labels = human.get("labels") or []
        draft_labels = draft.get("labels") or []
        counters["audited_frames"] += 1
        counters["human_instances"] += len(human_labels)
        counters["draft_instances_on_audited_frames"] += len(draft_labels)
        if not human_labels:
            counters["human_empty_frames"] += 1
            counters["draft_instances_on_human_empty_frames"] += len(draft_labels)
            rows.append(
                {
                    "route": route,
                    "stream_id": stream,
                    "frame_id": frame_id,
                    "human_species": "",
                    "draft_species": "",
                    "draft_species_method": "",
                    "draft_confidence_tier": "",
                    "inventory_distance_m": "",
                    "inventory_angular_error_deg": "",
                    "inventory_classifier_agreement": "",
                    "classifier_top1": "",
                    "classifier_confidence": "",
                    "classifier_species_agreement": "",
                    "mask_iou": "",
                    "matched_iou20": False,
                    "matched_iou50": False,
                    "species_known": False,
                    "species_agreement": "",
                    "human_record": str(source_path),
                    "draft_record": str(draft_path),
                    "overlay": str(draft_dir / "overlays" / route / f"{draft_path.stem}.jpg"),
                    "note": f"human_empty_with_{len(draft_labels)}_draft_proposals",
                }
            )
            continue

        available = set(range(len(draft_labels)))
        for human_label in human_labels:
            pairs = sorted(
                ((iou(human_label["points"], draft_labels[index]["points"]), index) for index in available),
                reverse=True,
            )
            best_iou, best_index = pairs[0] if pairs else (0.0, -1)
            matched = best_iou >= 0.20
            if matched:
                available.remove(best_index)
                counters["matched_iou20"] += 1
            if best_iou >= 0.50:
                counters["matched_iou50"] += 1
            draft_label = draft_labels[best_index] if matched else None
            draft_species = draft_label["species"] if draft_label else ""
            inventory_match = (draft_label or {}).get("inventory_match") or {}
            classifier = (draft_label or {}).get("classifier_suggestion") or {}
            classifier_species = classifier.get("top1", "")
            human_species = human_label.get("species", "")
            known = draft_species not in UNKNOWN_NAMES
            agreement = bool(known and canonical(draft_species) == canonical(human_species))
            if matched and known:
                counters["matched_with_inventory_species"] += 1
                counters["inventory_species_agreement"] += int(agreement)
            classifier_agreement = bool(
                matched and classifier_species and canonical(classifier_species) == canonical(human_species)
            )
            if matched and classifier_species:
                counters["matched_with_classifier_species"] += 1
                counters["classifier_species_agreement"] += int(classifier_agreement)
            rows.append(
                {
                    "route": route,
                    "stream_id": stream,
                    "frame_id": frame_id,
                    "human_species": human_species,
                    "draft_species": draft_species,
                    "draft_species_method": (draft_label or {}).get("species_method", ""),
                    "draft_confidence_tier": (draft_label or {}).get("species_confidence_tier", ""),
                    "inventory_distance_m": inventory_match.get("distance_m", ""),
                    "inventory_angular_error_deg": inventory_match.get("angular_error_deg", ""),
                    "inventory_classifier_agreement": (draft_label or {}).get(
                        "inventory_classifier_agreement", ""
                    ),
                    "classifier_top1": classifier_species,
                    "classifier_confidence": classifier.get("confidence", ""),
                    "classifier_species_agreement": classifier_agreement if matched and classifier_species else "",
                    "mask_iou": round(best_iou, 6),
                    "matched_iou20": matched,
                    "matched_iou50": best_iou >= 0.50,
                    "species_known": known,
                    "species_agreement": agreement if known else "",
                    "human_record": str(source_path),
                    "draft_record": str(draft_path),
                    "overlay": str(draft_dir / "overlays" / route / f"{draft_path.stem}.jpg"),
                    "note": "",
                }
            )

    recall20 = counters["matched_iou20"] / counters["human_instances"] if counters["human_instances"] else 0.0
    recall50 = counters["matched_iou50"] / counters["human_instances"] if counters["human_instances"] else 0.0
    agreement = (
        counters["inventory_species_agreement"] / counters["matched_with_inventory_species"]
        if counters["matched_with_inventory_species"]
        else 0.0
    )
    classifier_agreement = (
        counters["classifier_species_agreement"] / counters["matched_with_classifier_species"]
        if counters["matched_with_classifier_species"]
        else 0.0
    )
    summary = {
        "status": "complete",
        **dict(counters),
        "human_instance_recall_iou20": recall20,
        "human_instance_recall_iou50": recall50,
        "inventory_species_agreement_on_matched_known_instances": agreement,
        "classifier_species_agreement_on_matched_instances": classifier_agreement,
        "interpretation": {
            "iou20": "proposal overlaps enough to be useful as an editable draft",
            "iou50": "proposal is a reasonably aligned instance mask",
            "species_agreement": "draft government-inventory species equals prior human species after Latin-name normalization",
        },
    }
    (draft_dir / "human_calibration_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (draft_dir / "human_calibration_details.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["route"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
