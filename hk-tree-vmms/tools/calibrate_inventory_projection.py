#!/usr/bin/env python3
"""Estimate per-stream panorama yaw/sign from human species and inventory points."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_exhaustive_species_drafts import canonical, load_human_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COORDINATES = PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv"
INVENTORY = PROJECT_ROOT / "outputs" / "coordinates" / "nearby_trees.csv"
UNKNOWN = {"", "unknown / 待定", "unknown 未知"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def signed(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-range-m", type=float, default=45.0)
    args = parser.parse_args()
    coords = {(row["stream_id"], row["frame_id"]): row for row in rows(COORDINATES)}
    trees_by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
    for tree in rows(INVENTORY):
        trees_by_species[canonical(tree["species_name"])].append(tree)

    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (route, stream, frame_id), (record, path) in load_human_records().items():
        coordinate = coords.get((stream, frame_id))
        if coordinate is None:
            continue
        camera_x = float(coordinate["hk80_easting"])
        camera_y = float(coordinate["hk80_northing"])
        heading = float(coordinate["heading_deg"])
        for label in record.get("labels") or []:
            name = canonical(label.get("species", ""))
            if name in UNKNOWN:
                continue
            candidates: list[dict[str, float]] = []
            for tree in trees_by_species.get(name, []):
                dx = float(tree["hk80_easting"]) - camera_x
                dy = float(tree["hk80_northing"]) - camera_y
                distance = math.hypot(dx, dy)
                if 1.5 <= distance <= args.max_range_m:
                    bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                    candidates.append({"relative": signed(bearing - heading), "distance": distance})
            if not candidates:
                continue
            xs = [float(point[0]) for point in label["points"]]
            observations[stream].append(
                {
                    "target_angle": signed(((min(xs) + max(xs)) / 2.0 - 0.5) * 360.0),
                    "half_width": max(4.0, (max(xs) - min(xs)) * 180.0),
                    "candidates": candidates,
                    "species": label.get("species", ""),
                    "frame_id": frame_id,
                }
            )

    output: dict[str, Any] = {}
    for stream, items in observations.items():
        trials: list[dict[str, Any]] = []
        for sign in (1.0, -1.0):
            for offset in range(-180, 181):
                errors: list[float] = []
                inside = 0
                within10 = 0
                for item in items:
                    error = min(
                        abs(signed(sign * candidate["relative"] + offset - item["target_angle"]))
                        for candidate in item["candidates"]
                    )
                    errors.append(error)
                    inside += int(error <= item["half_width"])
                    within10 += int(error <= 10.0)
                ordered = sorted(errors)
                median = ordered[len(ordered) // 2]
                trials.append(
                    {
                        "sign": sign,
                        "yaw_offset_deg": offset,
                        "inside_human_bbox": inside,
                        "within_10deg": within10,
                        "median_error_deg": median,
                    }
                )
        best = sorted(trials, key=lambda trial: (-trial["inside_human_bbox"], -trial["within_10deg"], trial["median_error_deg"]))[0]
        output[stream] = {"usable_human_instances": len(items), "best": best, "current_default": next((trial for trial in trials if trial["sign"] == 1 and trial["yaw_offset_deg"] == 0), None)}
    args.output.resolve().write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
