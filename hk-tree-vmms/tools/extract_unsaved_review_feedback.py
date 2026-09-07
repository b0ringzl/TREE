#!/usr/bin/env python3
"""Extract explicit human edits from autosaved drafts without submitting them.

Only deltas against the pre-review draft snapshot are considered supervision.
Untouched model labels in a visited frame are not promoted to ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELER = PROJECT_ROOT / "derived/exhaustive_species_frame_labeler"
DEFAULT_DRAFTS = LABELER / "runtime/frame_drafts.json"
DEFAULT_BASELINE = PROJECT_ROOT / "backups/species_15m_reprocess_20260906_210228/frame_drafts.json"
DEFAULT_REVIEWS = LABELER / "runtime/frame_review_state.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "derived/unsaved_review_feedback_iteration2_20260906"
BANYAN = "榕树"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-utc", default="2026-09-06T13:04:08+00:00")
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_species(value: object) -> str:
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


def bbox(label: dict[str, Any]) -> tuple[float, float, float, float]:
    box = label.get("bbox") or {}
    if {"x_center", "y_center", "width", "height"} <= set(box):
        return (
            float(box["x_center"]) - float(box["width"]) / 2,
            float(box["y_center"]) - float(box["height"]) / 2,
            float(box["x_center"]) + float(box["width"]) / 2,
            float(box["y_center"]) + float(box["height"]) / 2,
        )
    points = label.get("points") or []
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    if not xs:
        raise ValueError("Label has no usable geometry")
    return min(xs), min(ys), max(xs), max(ys)


def crop(image: Image.Image, label: dict[str, Any]) -> Image.Image:
    left, top, right, bottom = bbox(label)
    width, height = image.size
    pad_x = max(8, round((right - left) * width * 0.12))
    pad_y = max(8, round((bottom - top) * height * 0.12))
    return image.crop(
        (
            max(0, int(left * width) - pad_x),
            max(0, int(top * height) - pad_y),
            min(width, int(right * width + 0.999) + pad_x),
            min(height, int(bottom * height + 0.999) + pad_y),
        )
    ).convert("RGB")


def source_image(frame_key: str, reviews: dict[str, Any]) -> Path:
    record = reviews.get(frame_key) or {}
    direct = Path(str(record.get("source_image") or ""))
    if direct.is_file():
        return direct.resolve()
    stream_id, frame_id = frame_key.rsplit("__", 1)
    cached = LABELER / "preview_cache" / stream_id / f"{frame_id}.jpg"
    if cached.is_file():
        return cached.resolve()
    raise FileNotFoundError(f"No source image or cached preview for {frame_key}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = parse_args()
    output = cfg.output.resolve()
    safe_reset(output, cfg.overwrite)
    drafts = read_json(cfg.drafts.resolve())
    baseline = read_json(cfg.baseline.resolve())
    reviews = read_json(cfg.reviews.resolve())
    touched = {
        key: value
        for key, value in drafts.items()
        if str(value.get("updated_at_utc") or "") > cfg.cutoff_utc
    }

    totals = Counter()
    relabels = Counter()
    frames = []
    species_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []

    for frame_key, current in sorted(
        touched.items(), key=lambda item: item[1].get("updated_at_utc", "")
    ):
        previous = baseline.get(frame_key) or {}
        old_labels = {
            label.get("label_id"): label
            for label in previous.get("labels") or []
            if label.get("label_id")
        }
        new_labels = {
            label.get("label_id"): label
            for label in current.get("labels") or []
            if label.get("label_id")
        }
        common = set(old_labels) & set(new_labels)
        changed_species = [
            key
            for key in common
            if normalize_species(old_labels[key].get("species"))
            != normalize_species(new_labels[key].get("species"))
        ]
        changed_geometry = [
            key
            for key in common
            if old_labels[key].get("points") != new_labels[key].get("points")
        ]
        added = sorted(set(new_labels) - set(old_labels))
        deleted = sorted(set(old_labels) - set(new_labels))
        status_changed = current.get("frame_status") != previous.get("frame_status")
        if not (changed_species or changed_geometry or added or deleted or status_changed):
            continue

        with Image.open(source_image(frame_key, reviews)) as opened:
            image = opened.convert("RGB")
            for label_id in changed_species + added:
                label = new_labels[label_id]
                species = normalize_species(label.get("species"))
                kind = "draft_relabel" if label_id in changed_species else "draft_added"
                target = output / "species_crops" / f"{frame_key}__{label_id}__{kind}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                crop(image, label).save(target, quality=94, optimize=True)
                species_rows.append(
                    {
                        "frame_key": frame_key,
                        "stream_id": frame_key.rsplit("__", 1)[0],
                        "updated_at_utc": current.get("updated_at_utc", ""),
                        "source_kind": kind,
                        "label_id": label_id,
                        "species": species,
                        "crop": str(target.resolve()),
                    }
                )
            for label_id in deleted:
                label = old_labels[label_id]
                target = output / "binary_negative_crops" / f"{frame_key}__{label_id}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                crop(image, label).save(target, quality=94, optimize=True)
                negative_rows.append(
                    {
                        "frame_key": frame_key,
                        "stream_id": frame_key.rsplit("__", 1)[0],
                        "updated_at_utc": current.get("updated_at_utc", ""),
                        "label_id": label_id,
                        "old_species": normalize_species(label.get("species")),
                        "crop": str(target.resolve()),
                    }
                )
            for label_id in sorted(set(changed_geometry) | set(added)):
                label = new_labels[label_id]
                target = output / "detector_positive_crops" / f"{frame_key}__{label_id}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                crop(image, label).save(target, quality=94, optimize=True)
                positive_rows.append(
                    {
                        "frame_key": frame_key,
                        "stream_id": frame_key.rsplit("__", 1)[0],
                        "updated_at_utc": current.get("updated_at_utc", ""),
                        "label_id": label_id,
                        "species": normalize_species(label.get("species")),
                        "source_kind": "geometry_edit" if label_id in changed_geometry else "draft_added",
                        "crop": str(target.resolve()),
                    }
                )

        for label_id in changed_species:
            source = normalize_species(old_labels[label_id].get("species"))
            target = normalize_species(new_labels[label_id].get("species"))
            relabels[f"{source} -> {target}"] += 1
        totals["changed_frames"] += 1
        totals["relabels"] += len(changed_species)
        totals["geometry_edits"] += len(changed_geometry)
        totals["added"] += len(added)
        totals["deleted"] += len(deleted)
        totals["status_changed"] += int(status_changed)
        frames.append(
            {
                "frame_key": frame_key,
                "updated_at_utc": current.get("updated_at_utc"),
                "relabels": [
                    {
                        "from": normalize_species(old_labels[label_id].get("species")),
                        "to": normalize_species(new_labels[label_id].get("species")),
                    }
                    for label_id in changed_species
                ],
                "geometry_edits": len(changed_geometry),
                "added": [normalize_species(new_labels[label_id].get("species")) for label_id in added],
                "deleted": [normalize_species(old_labels[label_id].get("species")) for label_id in deleted],
                "status_changed": status_changed,
            }
        )

    write_csv(output / "species_feedback.csv", species_rows)
    write_csv(output / "binary_negative_feedback.csv", negative_rows)
    write_csv(output / "detector_positive_feedback.csv", positive_rows)
    report = {
        "status": "extracted_without_submitting_or_mutating_drafts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_utc": cfg.cutoff_utc,
        "autosaved_touched_frames": len(touched),
        "autosaved_touched_instances": sum(len(value.get("labels") or []) for value in touched.values()),
        "explicit_delta_totals": dict(totals),
        "relabel_confusion": dict(relabels),
        "species_feedback_samples": len(species_rows),
        "binary_negative_samples": len(negative_rows),
        "detector_positive_samples": len(positive_rows),
        "frames": frames,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
