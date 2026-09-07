#!/usr/bin/env python3
"""Build leakage-aware replay datasets from the second human review batch.

The class-agnostic detector receives submitted positive polygons and usable
``no_tree`` frames.  Unusable frames are deliberately excluded.  The species
head receives only labels already represented by its stable 15-class head;
new species are exported to a visual-memory bank instead of being forced into
an unvalidated one-shot neural class.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELER = PROJECT_ROOT / "derived" / "exhaustive_species_frame_labeler"
RUNTIME = LABELER / "runtime"
DEFAULT_REVIEW_STATE = RUNTIME / "frame_review_state.json"
DEFAULT_PRE_REVIEW = (
    PROJECT_ROOT
    / "backups/species_15m_reprocess_20260906_210228/frame_drafts.json"
)
DEFAULT_SPECIES_BASE = PROJECT_ROOT / "derived/review_replay_species_20260906"
DEFAULT_DETECTOR_BASE = (
    PROJECT_ROOT / "derived/combined_tree_segmentation/effective_yolo_20260901_103256"
)
DEFAULT_SPECIES_OUTPUT = PROJECT_ROOT / "derived/review_replay_species_iteration2_20260906"
DEFAULT_DETECTOR_OUTPUT = PROJECT_ROOT / "derived/tree_detector_iteration2_20260906"
DEFAULT_REPORT = PROJECT_ROOT / "reports/20260906_第二轮人工反馈学习数据.json"
DEFAULT_UNSAVED_FEEDBACK = (
    PROJECT_ROOT / "derived/unsaved_review_feedback_iteration2_20260906/species_feedback.csv"
)
DEFAULT_UNSAVED_SUMMARY = (
    PROJECT_ROOT / "derived/unsaved_review_feedback_iteration2_20260906/summary.json"
)
DEFAULT_DRAFTS = LABELER / "runtime/frame_drafts.json"
BANYAN = "榕树"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-utc", default="2026-09-06T13:04:08+00:00")
    parser.add_argument("--review-state", type=Path, default=DEFAULT_REVIEW_STATE)
    parser.add_argument("--pre-review", type=Path, default=DEFAULT_PRE_REVIEW)
    parser.add_argument("--species-base", type=Path, default=DEFAULT_SPECIES_BASE)
    parser.add_argument("--detector-base", type=Path, default=DEFAULT_DETECTOR_BASE)
    parser.add_argument("--species-output", type=Path, default=DEFAULT_SPECIES_OUTPUT)
    parser.add_argument("--detector-output", type=Path, default=DEFAULT_DETECTOR_OUTPUT)
    parser.add_argument("--unsaved-feedback", type=Path, default=DEFAULT_UNSAVED_FEEDBACK)
    parser.add_argument("--unsaved-summary", type=Path, default=DEFAULT_UNSAVED_SUMMARY)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--species-repeat", type=int, default=8)
    parser.add_argument("--unsaved-species-repeat", type=int, default=4)
    parser.add_argument("--detector-repeat", type=int, default=12)
    parser.add_argument("--unsaved-detector-repeat", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def class_slug(index: int, species: str) -> str:
    if species == BANYAN:
        stem = "Ficus_banyan"
    else:
        latin = "_".join(species.split("(", 1)[0].split()[:2])
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", latin).strip("_") or "species"
    return f"{index:02d}_{stem}"


def image_path(record: dict[str, Any]) -> Path:
    source = Path(str(record.get("source_image") or ""))
    if source.is_file():
        return source.resolve()
    frame = record.get("frame") or {}
    fallback = Path(str(frame.get("source_image") or ""))
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Source image missing for {record.get('frame_key')}: {source}")


def bbox_from_label(label: dict[str, Any]) -> tuple[float, float, float, float]:
    box = label.get("bbox") or {}
    if {"x_center", "y_center", "width", "height"} <= set(box):
        left = float(box["x_center"]) - float(box["width"]) / 2
        top = float(box["y_center"]) - float(box["height"]) / 2
        right = float(box["x_center"]) + float(box["width"]) / 2
        bottom = float(box["y_center"]) + float(box["height"]) / 2
        return left, top, right, bottom
    points = label.get("points") or []
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    if not xs:
        raise ValueError("Label has neither bbox nor points")
    return min(xs), min(ys), max(xs), max(ys)


def crop_label(image: Image.Image, label: dict[str, Any]) -> Image.Image:
    left, top, right, bottom = bbox_from_label(label)
    width, height = image.size
    pad_x = max(8, int((right - left) * width * 0.12))
    pad_y = max(8, int((bottom - top) * height * 0.12))
    pixels = (
        max(0, int(left * width) - pad_x),
        max(0, int(top * height) - pad_y),
        min(width, int(right * width + 0.999) + pad_x),
        min(height, int(bottom * height + 0.999) + pad_y),
    )
    return image.crop(pixels).convert("RGB")


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_species(
    cfg: argparse.Namespace,
    reviews: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = cfg.species_base.resolve()
    output = cfg.species_output.resolve()
    safe_reset(output, cfg.overwrite)
    base_summary = read_json(base / "summary.json")
    base_rows = read_csv(base / "manifest.csv")
    unsaved_rows = (
        read_csv(cfg.unsaved_feedback.resolve())
        if cfg.unsaved_feedback.resolve().is_file()
        else []
    )
    stable_classes = sorted(base_summary["class_mapping"], key=str.casefold)
    mapping = {name: class_slug(index, name) for index, name in enumerate(stable_classes)}
    reviewed_keys = {record["frame_key"] for record in reviews} | {
        row["frame_key"] for row in unsaved_rows
    }
    rows: list[dict[str, Any]] = []
    materialized = Counter()

    for row in base_rows:
        if row["frame_key"] in reviewed_keys:
            continue
        species = normalize_species(row["species"])
        if species not in mapping:
            continue
        source = Path(row["crop"]).resolve()
        filename = f"base_{len(rows):06d}_{source.name}"
        target = output / row["split"] / mapping[species] / filename
        materialized[link_or_copy(source, target)] += 1
        rows.append(
            {
                "split": row["split"],
                "species": species,
                "class_folder": mapping[species],
                "source_kind": row["source_kind"],
                "frame_key": row["frame_key"],
                "route_block_id": row["route_block_id"],
                "source_crop": str(source),
                "crop": str(target),
            }
        )

    memory_rows: list[dict[str, Any]] = []
    for record in reviews:
        if record.get("frame_status") != "annotated":
            continue
        with Image.open(image_path(record)) as opened:
            image = opened.convert("RGB")
            for index, label in enumerate(record.get("labels") or []):
                species = normalize_species(label.get("species"))
                if not species or species.casefold().startswith("unknown") or species == "待定":
                    continue
                crop = crop_label(image, label)
                memory_dir = output / "human_memory" / "crops"
                memory_dir.mkdir(parents=True, exist_ok=True)
                memory_path = memory_dir / f"{record['frame_key']}__{index:02d}.jpg"
                crop.save(memory_path, quality=94, optimize=True)
                item = {
                    "species": species,
                    "frame_key": record["frame_key"],
                    "stream_id": record.get("stream_id", ""),
                    "route_distance_m": (record.get("frame") or {}).get("route_distance_m"),
                    "route_block_id": record.get("route_block_id", ""),
                    "label_id": label.get("label_id", ""),
                    "bbox": json.dumps(label.get("bbox") or {}, ensure_ascii=False),
                    "crop": str(memory_path.resolve()),
                    "in_stable_head": species in mapping,
                    "source_kind": "submitted_human_review",
                }
                memory_rows.append(item)
                if species not in mapping:
                    continue
                for repeat in range(cfg.species_repeat):
                    target = (
                        output
                        / "train"
                        / mapping[species]
                        / f"human_{record['frame_key']}__{index:02d}__r{repeat:02d}.jpg"
                    )
                    materialized[link_or_copy(memory_path, target)] += 1
                    rows.append(
                        {
                            "split": "train",
                            "species": species,
                            "class_folder": mapping[species],
                            "source_kind": "iteration2_human_review",
                            "frame_key": record["frame_key"],
                            "route_block_id": record.get("route_block_id", ""),
                            "source_crop": str(memory_path),
                            "crop": str(target),
                        }
                    )

    for index, feedback in enumerate(unsaved_rows):
        species = normalize_species(feedback.get("species"))
        source = Path(feedback["crop"]).resolve()
        if not source.is_file() or not species:
            continue
        memory_target = (
            output
            / "human_memory/crops"
            / f"unsaved_{feedback['frame_key']}__{feedback['label_id']}__{index:03d}{source.suffix.lower()}"
        )
        link_or_copy(source, memory_target)
        item = {
            "species": species,
            "frame_key": feedback["frame_key"],
            "stream_id": feedback.get("stream_id", ""),
            "route_distance_m": "",
            "route_block_id": "",
            "label_id": feedback.get("label_id", ""),
            "bbox": "{}",
            "crop": str(memory_target.resolve()),
            "in_stable_head": species in mapping,
            "source_kind": feedback.get("source_kind", "unsaved_draft_delta"),
        }
        memory_rows.append(item)
        if species not in mapping:
            continue
        for repeat in range(cfg.unsaved_species_repeat):
            target = (
                output
                / "train"
                / mapping[species]
                / f"unsaved_{feedback['frame_key']}__{feedback['label_id']}__r{repeat:02d}.jpg"
            )
            materialized[link_or_copy(memory_target, target)] += 1
            rows.append(
                {
                    "split": "train",
                    "species": species,
                    "class_folder": mapping[species],
                    "source_kind": feedback.get("source_kind", "unsaved_draft_delta"),
                    "frame_key": feedback["frame_key"],
                    "route_block_id": "",
                    "source_crop": str(memory_target),
                    "crop": str(target),
                }
            )
    # Keep the class index identical between train and val, including rare dirs.
    for folder in mapping.values():
        (output / "train" / folder).mkdir(parents=True, exist_ok=True)
        (output / "val" / folder).mkdir(parents=True, exist_ok=True)
    write_manifest(output / "manifest.csv", rows)
    write_manifest(output / "human_memory/manifest.csv", memory_rows)
    counts = Counter((row["split"], row["species"]) for row in rows)
    memory_counts = Counter(row["species"] for row in memory_rows)
    summary = {
        "status": "ready",
        "policy": (
            "stable 15-class replay head; second-round submitted labels repeated only in train; "
            "novel species retained in a visual-memory sidecar; unusable/no-tree frames excluded from species"
        ),
        "class_mapping": mapping,
        "train_crops": sum(value for (split, _), value in counts.items() if split == "train"),
        "val_crops": sum(value for (split, _), value in counts.items() if split == "val"),
        "human_memory_crops": len(memory_rows),
        "unsaved_feedback_crops": len(unsaved_rows),
        "human_memory_per_species": dict(memory_counts),
        "novel_memory_species": sorted(
            [name for name in memory_counts if name not in mapping], key=str.casefold
        ),
        "materialized": dict(materialized),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, memory_rows


def build_detector(cfg: argparse.Namespace, reviews: list[dict[str, Any]]) -> dict[str, Any]:
    base = cfg.detector_base.resolve()
    output = cfg.detector_output.resolve()
    safe_reset(output, cfg.overwrite)
    base_rows = read_csv(base / "manifest.csv")
    drafts = read_json(cfg.drafts.resolve())
    unsaved_summary = (
        read_json(cfg.unsaved_summary.resolve())
        if cfg.unsaved_summary.resolve().is_file()
        else {}
    )
    unsaved_keys = {
        item["frame_key"] for item in unsaved_summary.get("frames", [])
    }
    reviewed_keys = {record["frame_key"] for record in reviews} | unsaved_keys
    rows: list[dict[str, Any]] = []
    materialized = Counter()

    for row in base_rows:
        if row["frame_key"] in reviewed_keys:
            continue
        image_source = Path(row["image"]).resolve()
        label_source = Path(row["label"]).resolve()
        filename = f"base_{len(rows):06d}_{image_source.name}"
        image_target = output / "images" / row["split"] / filename
        label_target = output / "labels" / row["split"] / f"{Path(filename).stem}.txt"
        materialized[link_or_copy(image_source, image_target)] += 1
        materialized[link_or_copy(label_source, label_target)] += 1
        rows.append(
            {
                "split": row["split"],
                "source_kind": "base_replay",
                "frame_key": row["frame_key"],
                "route_block_id": row["route_block_id"],
                "image": str(image_target),
                "label": str(label_target),
                "tree_count": row["tree_count"],
            }
        )

    added_status = Counter()
    for record in reviews:
        status = record.get("frame_status")
        if status == "unusable":
            added_status["excluded_unusable"] += 1
            continue
        if status not in {"annotated", "no_tree"}:
            continue
        labels = record.get("labels") or []
        source = image_path(record)
        lines = []
        for label in labels:
            points = label.get("points") or []
            if len(points) < 3:
                left, top, right, bottom = bbox_from_label(label)
                points = [(left, top), (right, top), (right, bottom), (left, bottom)]
            coords = " ".join(f"{max(0.0, min(1.0, float(value))):.6f}" for point in points for value in point)
            lines.append(f"0 {coords}")
        for repeat in range(cfg.detector_repeat):
            stem = f"human_{record['frame_key']}__r{repeat:02d}"
            image_target = output / "images/train" / f"{stem}{source.suffix.lower()}"
            label_target = output / "labels/train" / f"{stem}.txt"
            materialized[link_or_copy(source, image_target)] += 1
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            rows.append(
                {
                    "split": "train",
                    "source_kind": f"iteration2_human_{status}",
                    "frame_key": record["frame_key"],
                    "route_block_id": record.get("route_block_id", ""),
                    "image": str(image_target),
                    "label": str(label_target),
                    "tree_count": len(lines),
                }
            )
        added_status[status] += 1

    for frame_key in sorted(unsaved_keys):
        draft = drafts.get(frame_key)
        if not draft:
            continue
        stream_id, frame_id = frame_key.rsplit("__", 1)
        source = LABELER / "preview_cache" / stream_id / f"{frame_id}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"Missing preview for unsaved feedback frame: {source}")
        lines = []
        for label in draft.get("labels") or []:
            points = label.get("points") or []
            if len(points) < 3:
                left, top, right, bottom = bbox_from_label(label)
                points = [(left, top), (right, top), (right, bottom), (left, bottom)]
            coords = " ".join(
                f"{max(0.0, min(1.0, float(value))):.6f}"
                for point in points
                for value in point
            )
            lines.append(f"0 {coords}")
        for repeat in range(cfg.unsaved_detector_repeat):
            stem = f"unsaved_{frame_key}__r{repeat:02d}"
            image_target = output / "images/train" / f"{stem}.jpg"
            label_target = output / "labels/train" / f"{stem}.txt"
            materialized[link_or_copy(source, image_target)] += 1
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            rows.append(
                {
                    "split": "train",
                    "source_kind": "iteration2_unsaved_human_delta_frame",
                    "frame_key": frame_key,
                    "route_block_id": "",
                    "image": str(image_target),
                    "label": str(label_target),
                    "tree_count": len(lines),
                }
            )
        added_status["unsaved_delta_frame"] += 1

    write_manifest(output / "manifest.csv", rows)
    data_yaml = (
        f'path: "{output.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n  0: tree\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    counts = Counter(row["split"] for row in rows)
    summary = {
        "status": "ready",
        "policy": (
            "base route-block validation/test frozen; second-round annotated and no_tree frames train-only; "
            "unusable frames excluded; deleted proposals become background through complete frame supervision"
        ),
        "split_frames_materialized": dict(counts),
        "new_review_frames": dict(added_status),
        "detector_repeat": cfg.detector_repeat,
        "unsaved_detector_repeat": cfg.unsaved_detector_repeat,
        "materialized": dict(materialized),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def audit_feedback(
    reviews: list[dict[str, Any]], pre_review: dict[str, Any]
) -> dict[str, Any]:
    totals = Counter()
    changes = Counter()
    per_frame = []
    for record in reviews:
        frame_key = record["frame_key"]
        before = (pre_review.get(frame_key) or {}).get("labels") or []
        after = record.get("labels") or []
        old = {label.get("label_id"): label for label in before if label.get("label_id")}
        new = {label.get("label_id"): label for label in after if label.get("label_id")}
        shared = set(old) & set(new)
        relabels = [
            (normalize_species(old[key].get("species")), normalize_species(new[key].get("species")))
            for key in shared
            if normalize_species(old[key].get("species")) != normalize_species(new[key].get("species"))
        ]
        deleted = [normalize_species(old[key].get("species")) for key in set(old) - set(new)]
        added = [normalize_species(new[key].get("species")) for key in set(new) - set(old)]
        totals["reviewed_frames"] += 1
        totals[f"status/{record.get('frame_status')}"] += 1
        totals["before_instances"] += len(before)
        totals["after_instances"] += len(after)
        totals["deleted_instances"] += len(deleted)
        totals["added_instances"] += len(added)
        totals["relabeled_instances"] += len(relabels)
        for source, target in relabels:
            changes[f"{source} -> {target}"] += 1
        per_frame.append(
            {
                "frame_key": frame_key,
                "status": record.get("frame_status"),
                "before": len(before),
                "after": len(after),
                "deleted": deleted,
                "added": added,
                "relabeled": [{"from": source, "to": target} for source, target in relabels],
            }
        )
    return {"totals": dict(totals), "relabel_confusion": dict(changes), "frames": per_frame}


def main() -> None:
    cfg = parse_args()
    state = read_json(cfg.review_state.resolve())
    reviews = sorted(
        [
            record
            for record in state.values()
            if str(record.get("updated_at_utc") or "") > cfg.cutoff_utc
        ],
        key=lambda record: record["updated_at_utc"],
    )
    if not reviews:
        raise RuntimeError("No submitted reviews found after cutoff")
    pre_review = read_json(cfg.pre_review.resolve())
    feedback = audit_feedback(reviews, pre_review)
    species_summary, memory_rows = build_species(cfg, reviews)
    detector_summary = build_detector(cfg, reviews)
    report = {
        "status": "ready_for_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_utc": cfg.cutoff_utc,
        "feedback": feedback,
        "species_dataset": str(cfg.species_output.resolve()),
        "species": species_summary,
        "human_memory_manifest": str(
            (cfg.species_output.resolve() / "human_memory/manifest.csv")
        ),
        "human_memory_samples": len(memory_rows),
        "detector_dataset": str(cfg.detector_output.resolve()),
        "detector": detector_summary,
    }
    cfg.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    cfg.report.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
