#!/usr/bin/env python3
"""Create a deterministic random classification-review sample for TST and Stubbs Road."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "derived" / "exhaustive_species_review_20260906"
DEFAULT_LABELER = PROJECT_ROOT / "derived" / "exhaustive_species_frame_labeler"
DEFAULT_COORDINATES = PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "20260906_尖沙咀及司徒拔道随机分类验收样本.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labeler", type=Path, default=DEFAULT_LABELER)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-route", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def round_robin_blocks(records: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["route_block_id"]].append(record)
    for values in groups.values():
        rng.shuffle(values)
    block_names = list(groups)
    rng.shuffle(block_names)
    selected = []
    while len(selected) < count:
        changed = False
        for block in block_names:
            if groups[block] and len(selected) < count:
                selected.append(groups[block].pop())
                changed = True
        if not changed:
            break
    return selected


def main() -> None:
    cfg = parse_args()
    rng = random.Random(cfg.seed)
    # Match the review UI's 5 m anchor sequence exactly so the CSV can be used
    # directly with its numbered jump control.
    from evaluate_human_review_accuracy import sampled_frame_keys

    ui_indices = {
        key: index + 1
        for index, key in enumerate(sampled_frame_keys(cfg.coordinates.resolve(), 5.0))
    }
    draft_path = cfg.labeler.resolve() / "runtime" / "frame_drafts.json"
    drafts = json.loads(draft_path.read_text(encoding="utf-8"))
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((cfg.source.resolve() / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        route = record.get("dataset_route")
        if route not in {"jianshazui", "stubbs_road"}:
            continue
        if record.get("status") != "auto_review_required" or not record.get("labels"):
            continue
        frame_key = f"{record['stream_id']}__{record['frame_id']}"
        draft = drafts.get(frame_key) or {}
        distance = float((record.get("camera") or {}).get("route_distance_m") or 0.0)
        record["route_block_id"] = (
            draft.get("route_block_id")
            or f"{record['stream_id']}_{int(distance // 100):04d}"
        )
        record["_record_path"] = str(path)
        by_route[route].append(record)

    rows = []
    for route in ("jianshazui", "stubbs_road"):
        selected = round_robin_blocks(by_route[route], min(cfg.per_route, len(by_route[route])), rng)
        for route_index, record in enumerate(selected, start=1):
            rows.append(
                {
                    "sample_order": len(rows) + 1,
                    "route_sample_order": route_index,
                    "route": route,
                    "stream_id": record["stream_id"],
                    "frame_id": record["frame_id"],
                    "frame_key": f"{record['stream_id']}__{record['frame_id']}",
                    "ui_frame_number": ui_indices.get(
                        f"{record['stream_id']}__{record['frame_id']}", ""
                    ),
                    "route_block_id": record["route_block_id"],
                    "pre_review_instances": len(record.get("labels") or []),
                    "review_status": "待验收",
                    "source_record": record["_record_path"],
                }
            )
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "seed": cfg.seed,
        "per_route": cfg.per_route,
        "selection": "auto_review_required frames with at least one proposal; randomized round-robin over route blocks",
        "rows": len(rows),
        "route_counts": {route: sum(row["route"] == route for row in rows) for route in by_route},
        "holdout_blocks": sorted({row["route_block_id"] for row in rows}),
        "note": "Keep these frames out of future tuning until accuracy is finalized.",
    }
    cfg.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
