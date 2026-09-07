#!/usr/bin/env python3
"""Import the conservative species review package into the editable frame UI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANYAN_LABEL = "榕树"


DEFAULT_PREPROCESS = {
    "recipe_version": "photo_v1",
    "exposure_ev": 0.0,
    "contrast": 1.0,
    "reason": "normal",
    "auto_suggested": False,
    "human_accepted": False,
}


def normalize_species(value: object) -> str:
    name = str(value or "").strip()
    folded = name.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return BANYAN_LABEL
    if any(token in name for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return BANYAN_LABEL
    return BANYAN_LABEL if name == "榕樹" else name


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def label_note(label: dict[str, Any]) -> str:
    notes = [str(label.get("note", "")).strip()]
    if label.get("candidate_species"):
        notes.append(f"清单/初始候选：{normalize_species(label['candidate_species'])}")
    hint = label.get("vmms_domain_classifier_suggestion") or {}
    if hint.get("top1_species"):
        notes.append(f"域内YOLO候选：{normalize_species(hint['top1_species'])} ({float(hint.get('confidence', 0)):.3f})")
    inventory = label.get("inventory_match") or {}
    if inventory.get("source_tree_id"):
        notes.append(
            f"政府树号：{inventory['source_tree_id']}，距离 {inventory.get('distance_m', '?')} m，"
            f"模型/清单一致={hint.get('agrees_with_inventory', False)}"
        )
    return "；".join(value for value in notes if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    cfg = parser.parse_args()
    review, output = cfg.review_dir.resolve(), cfg.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing editable import: {output}")
    (output / "preview_cache").mkdir(parents=True)
    (output / "annotations").mkdir(parents=True)
    (output / "runtime").mkdir(parents=True)
    classes = list(
        dict.fromkeys(
            normalize_species(name)
            for name in json.loads((review / "classes.json").read_text(encoding="utf-8"))
        )
    )
    with (PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        coordinates = {
            (row["stream_id"], row["frame_id"]): row for row in csv.DictReader(stream)
        }
    (output / "annotations" / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    drafts: dict[str, Any] = {}
    reviews: dict[str, Any] = {}
    imported = {"draft_frames": 0, "accepted_status_frames": 0, "labels": 0}
    for record_path in sorted((review / "records").glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        stream_id, frame_id = record["stream_id"], str(record["frame_id"])
        frame_key = f"{stream_id}__{frame_id}"
        source_image = review / "images" / f"{record_path.stem}.jpg"
        link_or_copy(source_image, output / "preview_cache" / stream_id / f"{frame_id}.jpg")
        labels = []
        for label in record.get("labels") or []:
            label_species = normalize_species(label.get("species", "Unknown / 待定"))
            hint = label.get("vmms_domain_classifier_suggestion") or {}
            hint_species = normalize_species(hint.get("top1_species"))
            confidence = (
                float(hint["confidence"])
                if hint_species == label_species and hint.get("confidence") is not None
                else None
            )
            labels.append(
                {
                    "label_id": label["label_id"],
                    "species": label_species,
                    "points": label["points"],
                    "visibility": label.get(
                        "visibility",
                        "uncertain" if label.get("requires_human_review", True) else "clear",
                    ),
                    "note": label_note(label),
                    "confidence": confidence,
                }
            )
        status = record.get("status", "auto_review_required")
        coordinate = coordinates[(stream_id, frame_id)]
        camera = record["camera"]
        frame = {
            "frame_key": frame_key,
            "stream_id": stream_id,
            "frame_id": frame_id,
            "seq_id": int(coordinate["seq_id"]),
            "utc_datetime": coordinate["utc_datetime"],
            "hong_kong_datetime": coordinate["hong_kong_datetime"],
            "source_image_relpath": coordinate["source_image_relpath"],
            "easting": float(coordinate["hk80_easting"]),
            "northing": float(coordinate["hk80_northing"]),
            "latitude": float(coordinate["wgs84_latitude"]),
            "longitude": float(coordinate["wgs84_longitude"]),
            "heading_deg": float(coordinate["heading_deg"]) % 360.0,
            "route_distance_m": float(camera["route_distance_m"]),
        }
        frame_status = (
            "no_tree" if status == "human_verified_no_tree"
            else "unusable" if status == "excluded_human_unusable"
            else "annotated"
        )
        editable = {
            "frame_id": frame_key,
            "frame_key": frame_key,
            "stream_id": stream_id,
            "frame_status": frame_status,
            "labels": labels if frame_status == "annotated" else [],
            "draft_points": [],
            "preprocess": DEFAULT_PREPROCESS,
            "note": f"导入自穷尽式树种预标注；原状态={status}",
            "frame": frame,
            "image_size": record["image_size"],
            "source_image": str((PROJECT_ROOT.parent.parent / "vmms" / coordinate["source_image_relpath"]).resolve()),
            "route_block_id": f"{stream_id}_{int(float(camera['route_distance_m']) // 100):04d}",
        }
        if frame_status in {"no_tree", "unusable"}:
            reviews[frame_key] = editable
            imported["accepted_status_frames"] += 1
        else:
            drafts[frame_key] = editable
            imported["draft_frames"] += 1
            imported["labels"] += len(labels)

    (output / "runtime" / "frame_drafts.json").write_text(
        json.dumps(drafts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "runtime" / "frame_review_state.json").write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "runtime" / "session.json").write_text(
        json.dumps({"spacing_m": 5.0, "current_index": 0}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "ready_for_editable_review",
        "source_review": str(review),
        "output_root": str(output),
        **imported,
        "instructions": "Launch 启动VMMS树种预标注验收工具.cmd, select a crown, change species, then save and continue.",
    }
    (output / "IMPORT_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
