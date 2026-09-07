#!/usr/bin/env python3
"""Replay the stable binary tree gate with explicit second-round draft deltas."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "derived/hwt_false_positive_filter_20260906"
DEFAULT_FEEDBACK = PROJECT_ROOT / "derived/unsaved_review_feedback_iteration2_20260906"
DEFAULT_FORMAL_MEMORY = (
    PROJECT_ROOT
    / "derived/review_replay_species_iteration2_full_20260906/human_memory/crops"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "derived/binary_tree_filter_iteration2_20260906"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--formal-memory", type=Path, default=DEFAULT_FORMAL_MEMORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--positive-repeat", type=int, default=3)
    parser.add_argument("--negative-repeat", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_reset(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    derived = (PROJECT_ROOT / "derived").resolve()
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {resolved}")
        if derived not in resolved.parents:
            raise ValueError(f"Refusing to replace output outside {derived}: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def images(path: Path) -> list[Path]:
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def main() -> None:
    cfg = parse_args()
    base = cfg.base.resolve()
    feedback = cfg.feedback.resolve()
    output = cfg.output.resolve()
    safe_reset(output, cfg.overwrite)
    materialized = Counter()
    counts = Counter()

    for split in ("train", "val"):
        for class_name in ("0_tree", "1_not_tree"):
            for index, source in enumerate(images(base / split / class_name)):
                target = output / split / class_name / f"base_{index:06d}_{source.name}"
                materialized[link_or_copy(source, target)] += 1
                counts[f"{split}/{class_name}"] += 1

    positives = images(feedback / "species_crops")
    formal = [
        path
        for path in images(cfg.formal_memory.resolve())
        if not path.name.startswith("unsaved_")
    ]
    negatives = images(feedback / "binary_negative_crops")
    for index, source in enumerate(positives + formal):
        for repeat in range(cfg.positive_repeat):
            target = output / "train/0_tree" / f"feedback_{index:04d}_r{repeat:02d}_{source.name}"
            materialized[link_or_copy(source, target)] += 1
            counts["train/0_tree"] += 1
    for index, source in enumerate(negatives):
        for repeat in range(cfg.negative_repeat):
            target = output / "train/1_not_tree" / f"feedback_{index:04d}_r{repeat:02d}_{source.name}"
            materialized[link_or_copy(source, target)] += 1
            counts["train/1_not_tree"] += 1

    summary = {
        "status": "ready",
        "policy": (
            "frozen binary validation; positive human relabel/addition crops replayed; "
            "explicit deletions replayed as not-tree; unusable frames excluded"
        ),
        "counts": dict(counts),
        "feedback": {
            "positive_unique": len(positives) + len(formal),
            "negative_unique": len(negatives),
            "positive_repeat": cfg.positive_repeat,
            "negative_repeat": cfg.negative_repeat,
        },
        "materialized": dict(materialized),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
