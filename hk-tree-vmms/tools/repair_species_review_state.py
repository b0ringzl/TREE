#!/usr/bin/env python3
"""Repair frame metadata in an already imported editable species review state."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler-root", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    cfg = parser.parse_args()
    root, review = cfg.labeler_root.resolve(), cfg.review_dir.resolve()
    review_path = root / "runtime" / "frame_review_state.json"
    drafts_path = root / "runtime" / "frame_drafts.json"
    state = json.loads(review_path.read_text(encoding="utf-8"))
    drafts = json.loads(drafts_path.read_text(encoding="utf-8"))
    with (PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        coordinates = {(row["stream_id"], row["frame_id"]): row for row in csv.DictReader(stream)}
    source_records = {}
    for path in (review / "records").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        source_records[(record["stream_id"], str(record["frame_id"]))] = record

    repaired = 0
    for frame_key, record in state.items():
        if record.get("frame"):
            continue
        stream_id, frame_id = frame_key.rsplit("__", 1)
        coordinate = coordinates[(stream_id, frame_id)]
        source = source_records[(stream_id, frame_id)]
        camera = source["camera"]
        record["frame"] = {
            "frame_key": frame_key, "stream_id": stream_id, "frame_id": frame_id,
            "seq_id": int(coordinate["seq_id"]), "utc_datetime": coordinate["utc_datetime"],
            "hong_kong_datetime": coordinate["hong_kong_datetime"],
            "source_image_relpath": coordinate["source_image_relpath"],
            "easting": float(coordinate["hk80_easting"]), "northing": float(coordinate["hk80_northing"]),
            "latitude": float(coordinate["wgs84_latitude"]), "longitude": float(coordinate["wgs84_longitude"]),
            "heading_deg": float(coordinate["heading_deg"]) % 360.0,
            "route_distance_m": float(camera["route_distance_m"]),
        }
        record["image_size"] = source["image_size"]
        record["source_image"] = str((PROJECT_ROOT.parent.parent / "vmms" / coordinate["source_image_relpath"]).resolve())
        record["route_block_id"] = f"{stream_id}_{int(float(camera['route_distance_m']) // 100):04d}"
        repaired += 1
    # A failed post-submit manifest rebuild may leave an auto-saved stale draft.
    stale = [key for key in drafts if key in state and state[key].get("updated_at_utc")]
    for key in stale:
        drafts.pop(key, None)
    atomic_json(review_path, state)
    atomic_json(drafts_path, drafts)
    print(json.dumps({"repaired_review_records": repaired, "removed_post_submit_drafts": stale}, ensure_ascii=False))


if __name__ == "__main__":
    main()
