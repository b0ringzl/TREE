#!/usr/bin/env python3
"""Build a species-only replay dataset from VMMS anchors and human corrections.

The previous refinement experiment put deleted detector proposals in the same
classification head as tree species.  This builder deliberately excludes those
hard negatives: they belong to the binary tree gate.  It mixes the original
cross-route VMMS crops with corrected Ho Man Tin crops so that adaptation does
not catastrophically forget Tsim Sha Tsui and Stubbs Road.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "derived" / "vmms_species_cls_20260906"
DEFAULT_REVIEW = PROJECT_ROOT / "derived" / "hwt_species_refinement_20260906"
DEFAULT_OUTPUT = PROJECT_ROOT / "derived" / "review_replay_species_20260906"
NEGATIVE = "非树/误检"
BANYAN = "榕树"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--review-data", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-train-per-class", type=int, default=96)
    parser.add_argument("--maximum-train-per-class", type=int, default=360)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_species(value: object) -> str:
    name = str(value or "").strip()
    folded = name.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return BANYAN
    if any(token in name for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return BANYAN
    return BANYAN if name == "榕樹" else name


def class_slug(index: int, species: str) -> str:
    if species == BANYAN:
        stem = "Ficus_banyan"
    else:
        latin = "_".join(species.split("(", 1)[0].split()[:2])
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", latin).strip("_") or "species"
    return f"{index:02d}_{stem}"


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    cfg = parse_args()
    base = cfg.base_data.resolve()
    review = cfg.review_data.resolve()
    output = cfg.output.resolve()
    if output.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        if PROJECT_ROOT.resolve() not in output.parents:
            raise ValueError(f"Refusing to replace output outside project: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    base_rows = read_rows(base / "manifest.csv")
    review_rows = read_rows(review / "manifest.csv")
    review_summary = json.loads((review / "summary.json").read_text(encoding="utf-8"))
    validation_blocks = set(review_summary["validation_blocks"])
    reviewed_frames = {row["frame_key"] for row in review_rows}

    samples: list[dict[str, Any]] = []
    excluded = Counter()
    for row in base_rows:
        species = normalize_species(row["species"])
        frame_key = f"{row['stream_id']}__{row['frame_id']}"
        # A submitted correction supersedes every older crop from the same frame.
        if frame_key in reviewed_frames:
            excluded["superseded_base_crop"] += 1
            continue
        source = Path(row["crop"])
        if not source.is_file():
            excluded["missing_base_crop"] += 1
            continue
        # Preserve the original held-out assignment for cross-route knowledge.
        split = "val" if row["split"] == "val" else "train"
        samples.append(
            {
                "split": split,
                "species": species,
                "source": source,
                "source_kind": "original_vmms_anchor",
                "frame_key": frame_key,
                "route_block_id": row["route_block_id"],
            }
        )

    for row in review_rows:
        species = normalize_species(row["species"])
        if species == NEGATIVE:
            excluded["binary_gate_negative"] += 1
            continue
        if "balanced_repeat" in row.get("source_kind", ""):
            excluded["old_balanced_repeat"] += 1
            continue
        source = Path(row["crop"])
        if not source.is_file():
            excluded["missing_review_crop"] += 1
            continue
        # The existing review builder used a whole-100 m-block holdout.  Keep it
        # unchanged so the old and new classifiers can be compared fairly.
        split = "val" if row["route_block_id"] in validation_blocks else "train"
        samples.append(
            {
                "split": split,
                "species": species,
                "source": source,
                "source_kind": "human_review_correction",
                "frame_key": row["frame_key"],
                "route_block_id": row["route_block_id"],
            }
        )

    class_names = sorted({sample["species"] for sample in samples}, key=str.casefold)
    mapping = {name: class_slug(index, name) for index, name in enumerate(class_names)}
    train_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    val_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        (train_by_class if sample["split"] == "train" else val_by_class)[sample["species"]].append(sample)

    rows: list[dict[str, Any]] = []
    materialized = Counter()

    def emit(sample: dict[str, Any], split: str, ordinal: int, repeat: int = 0) -> None:
        source = sample["source"]
        suffix = f"__repeat{repeat:04d}" if repeat else ""
        filename = f"{ordinal:06d}_{source.stem}{suffix}{source.suffix.lower()}"
        target = output / split / mapping[sample["species"]] / filename
        materialized[link_or_copy(source, target)] += 1
        rows.append(
            {
                "split": split,
                "species": sample["species"],
                "class_folder": mapping[sample["species"]],
                "source_kind": sample["source_kind"] + ("_balanced_repeat" if repeat else ""),
                "frame_key": sample["frame_key"],
                "route_block_id": sample["route_block_id"],
                "source_crop": str(source),
                "crop": str(target),
            }
        )

    ordinal = 0
    for species in class_names:
        originals = train_by_class[species]
        if not originals:
            continue
        # Cap dominant classes and deterministically repeat rare classes.  Repeat
        # links still receive independent stochastic augmentation during training.
        selected = originals[: cfg.maximum_train_per_class]
        target_count = min(
            cfg.maximum_train_per_class,
            max(cfg.minimum_train_per_class, len(selected)),
        )
        for sample in selected:
            emit(sample, "train", ordinal)
            ordinal += 1
        repeat = 1
        while len(selected) + repeat - 1 < target_count:
            sample = selected[(repeat - 1) % len(selected)]
            emit(sample, "train", ordinal, repeat)
            ordinal += 1
            repeat += 1
        for sample in val_by_class[species]:
            emit(sample, "val", ordinal)
            ordinal += 1

    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["species"]) for row in rows)
    summary = {
        "status": "ready",
        "policy": (
            "species-only head; deleted proposals are excluded for the binary gate; "
            "original VMMS anchors replayed; submitted corrections supersede old crops; "
            "Ho Man Tin validation remains separated by whole 100 m blocks"
        ),
        "class_mapping": mapping,
        "classes": len(mapping),
        "train_crops": sum(value for (split, _), value in counts.items() if split == "train"),
        "val_crops": sum(value for (split, _), value in counts.items() if split == "val"),
        "validation_blocks": sorted(validation_blocks),
        "per_class": {
            species: {
                "train": counts[("train", species)],
                "val": counts[("val", species)],
                "human_review_train": sum(
                    row["split"] == "train"
                    and row["species"] == species
                    and row["source_kind"].startswith("human_review_correction")
                    for row in rows
                ),
            }
            for species in class_names
        },
        "excluded": dict(excluded),
        "materialized": dict(materialized),
        "ficus_policy": "Ficus microcarpa and Ficus benjamina are one 榕树 class",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
