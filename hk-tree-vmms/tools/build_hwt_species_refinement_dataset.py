#!/usr/bin/env python3
"""Build a leakage-safe Ho Man Tin species refinement dataset.

The authoritative positives are the records submitted through the exhaustive
review UI.  Pre-review proposals that disappeared during review are retained as
hard negatives.  Ficus microcarpa and Ficus benjamina are merged into 榕树.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELER = PROJECT_ROOT / "derived" / "exhaustive_species_frame_labeler"
DEFAULT_SOURCE = PROJECT_ROOT / "derived" / "exhaustive_species_review_20260906"
DEFAULT_OUTCOMES = (
    PROJECT_ROOT / "reports" / "20260906_VMMS识别精度报告_人工验收1-280帧_逐实例.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "derived" / "hwt_species_refinement_20260906"
UNKNOWN = {"", "Unknown / 待定", "Unknown 未知"}
BANYAN = "榕树"
NEGATIVE = "非树/误检"
LEGACY_FOLDERS = {
    "Aleurites moluccana 石栗": "01_Aleurites_moluccana",
    "Alstonia scholaris 糖膠樹": "02_Alstonia_scholaris",
    "Araucaria heterophylla 异叶南洋杉": "03_Araucaria_heterophylla",
    "Bombax ceiba 木棉": "05_Bombax_ceiba",
    "Celtis sinensis 朴樹": "06_Celtis_sinensis",
    "Delonix regia 鳳凰木": "08_Delonix_regia",
    "Livistona chinensis 蒲葵": "11_Livistona_chinensis",
    "Schefflera heptaphylla 鵝掌柴(鴨腳木)": "12_Schefflera_heptaphylla",
    "Wodyetia bifurcata 狐尾椰子": "14_Wodyetia_bifurcata",
    "Phoenix roebelenii 日本葵(軟葉刺葵)": "15_Phoenix_roebelenii",
    BANYAN: "16_Ficus_banyan",
    NEGATIVE: "17_not_tree_false_positive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", type=Path, default=DEFAULT_LABELER)
    parser.add_argument("--source-review", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-positive-instances", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260906)
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
    if species in LEGACY_FOLDERS:
        return LEGACY_FOLDERS[species]
    if species == NEGATIVE:
        stem = "not_tree_false_positive"
    elif species == BANYAN:
        stem = "Ficus_banyan"
    else:
        latin = "_".join(species.split("(", 1)[0].split()[:2])
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", latin).strip("_") or "species"
    return f"{index:02d}_{stem}"


def box_from_label(label: dict[str, Any]) -> tuple[float, float, float, float]:
    box = label.get("bbox") or {}
    if all(key in box for key in ("left", "top", "right", "bottom")):
        return tuple(float(box[key]) for key in ("left", "top", "right", "bottom"))
    points = label.get("points") or []
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def padded_crop(image: Image.Image, label: dict[str, Any], padding: float) -> Image.Image | None:
    left, top, right, bottom = box_from_label(label)
    width, height = image.size
    x1, x2, y1, y2 = left * width, right * width, top * height, bottom * height
    pad = max(x2 - x1, y2 - y1) * padding
    bounds = (
        max(0, int(x1 - pad)),
        max(0, int(y1 - pad)),
        min(width, int(x2 + pad + 1)),
        min(height, int(y2 + pad + 1)),
    )
    if bounds[2] - bounds[0] < 16 or bounds[3] - bounds[1] < 16:
        return None
    return image.crop(bounds).convert("RGB")


def stable_score(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_source_records(source: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in (source / "records").glob("hewentian__hewentian_pano__*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        result[f"hewentian_pano__{record['frame_id']}"] = record
    return result


def route_block(record: dict[str, Any], source_record: dict[str, Any]) -> str:
    existing = record.get("route_block_id") or source_record.get("route_block_id")
    if existing:
        return str(existing)
    distance = float((source_record.get("camera") or {}).get("route_distance_m") or 0.0)
    return f"hewentian_pano_{int(distance // 100):04d}"


def choose_validation_blocks(
    samples: list[dict[str, Any]], val_fraction: float, seed: int
) -> set[str]:
    blocks = sorted({sample["route_block_id"] for sample in samples})
    target = max(1, round(len(blocks) * val_fraction))
    chosen = set(sorted(blocks, key=lambda value: stable_score(seed, value))[:target])
    by_species: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        by_species[sample["species"]].add(sample["route_block_id"])
    # Ensure every sufficiently distributed class is measurable without putting
    # all of that class's blocks into validation.
    for species, species_blocks in sorted(by_species.items()):
        if len(species_blocks) < 2 or species_blocks & chosen:
            continue
        candidate = min(species_blocks, key=lambda value: stable_score(seed + 1, value))
        chosen.add(candidate)
    for species, species_blocks in sorted(by_species.items()):
        if species_blocks and species_blocks <= chosen and len(species_blocks) > 1:
            chosen.remove(max(species_blocks, key=lambda value: stable_score(seed + 2, value)))
    return chosen


def main() -> None:
    cfg = parse_args()
    labeler = cfg.labeler.resolve()
    source = cfg.source_review.resolve()
    output = cfg.output.resolve()
    if output.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_records = load_source_records(source)
    positives: list[dict[str, Any]] = []
    submitted_frames: set[str] = set()
    submitted_root = labeler / "annotations" / "records" / "hewentian_pano"
    for path in sorted(submitted_root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        frame_id = str(record.get("frame_id") or path.stem)
        frame_key = f"hewentian_pano__{frame_id}"
        submitted_frames.add(frame_key)
        source_record = source_records.get(frame_key)
        if source_record is None:
            continue
        block = route_block(record, source_record)
        for index, label in enumerate(record.get("labels") or []):
            species = normalize_species(label.get("species"))
            if species in UNKNOWN:
                continue
            positives.append(
                {
                    "species": species,
                    "frame_key": frame_key,
                    "frame_id": frame_id,
                    "route_block_id": block,
                    "label_id": str(label.get("label_id") or f"human_{index}"),
                    "label": label,
                    "source_kind": "human_review_positive",
                }
            )

    positive_counts = Counter(sample["species"] for sample in positives)
    selected_species = {
        species for species, count in positive_counts.items() if count >= cfg.min_positive_instances
    }
    positives = [sample for sample in positives if sample["species"] in selected_species]

    deleted_ids: dict[str, set[str]] = defaultdict(set)
    with cfg.outcomes.resolve().open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("outcome") == "deleted":
                deleted_ids[row["frame_key"]].add(row["label_id"])
    negatives: list[dict[str, Any]] = []
    for frame_key, ids in deleted_ids.items():
        if frame_key not in submitted_frames:
            continue
        record = source_records.get(frame_key)
        if record is None:
            continue
        block = route_block({}, record)
        for label in record.get("labels") or []:
            label_id = str(label.get("label_id") or "")
            if label_id in ids:
                negatives.append(
                    {
                        "species": NEGATIVE,
                        "frame_key": frame_key,
                        "frame_id": record["frame_id"],
                        "route_block_id": block,
                        "label_id": label_id,
                        "label": label,
                        "source_kind": "deleted_hard_negative",
                    }
                )

    samples = positives + negatives
    val_blocks = choose_validation_blocks(samples, cfg.val_fraction, cfg.seed)
    class_names = sorted(selected_species, key=str.casefold) + [NEGATIVE]
    class_mapping = {species: class_slug(index, species) for index, species in enumerate(class_names)}
    rows: list[dict[str, Any]] = []
    base_train_counts: Counter[str] = Counter()

    opened: dict[str, Image.Image] = {}
    try:
        for sample in samples:
            frame_key = sample["frame_key"]
            image = opened.get(frame_key)
            if image is None:
                image_path = source / "images" / f"hewentian__{frame_key}.jpg"
                with Image.open(image_path) as raw:
                    image = raw.convert("RGB")
                opened[frame_key] = image
            split = "val" if sample["route_block_id"] in val_blocks else "train"
            paddings = (0.15,) if split == "val" else (0.10, 0.30)
            for variant, padding in enumerate(paddings):
                crop = padded_crop(image, sample["label"], padding)
                if crop is None:
                    continue
                filename = f"{frame_key}__{sample['label_id']}__p{variant}.jpg"
                target = output / split / class_mapping[sample["species"]] / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                crop.save(target, quality=94, optimize=True)
                rows.append(
                    {
                        "split": split,
                        "species": sample["species"],
                        "class_folder": class_mapping[sample["species"]],
                        "frame_key": frame_key,
                        "route_block_id": sample["route_block_id"],
                        "label_id": sample["label_id"],
                        "source_kind": sample["source_kind"],
                        "padding": padding,
                        "crop": str(target),
                    }
                )
                if split == "train":
                    base_train_counts[sample["species"]] += 1
    finally:
        for image in opened.values():
            image.close()

    # Balance under-represented classes by hard-linking deterministic repeats.
    target_per_class = min(120, max(40, int(sorted(base_train_counts.values())[len(base_train_counts) // 2])))
    by_species_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_species_rows[row["species"]].append(row)
    for species, species_rows in sorted(by_species_rows.items()):
        if not species_rows:
            continue
        originals = list(species_rows)
        repeat = 0
        current_count = len(originals)
        while current_count < target_per_class:
            source_row = originals[repeat % len(originals)]
            source_path = Path(source_row["crop"])
            target = source_path.with_name(f"{source_path.stem}__repeat{repeat:03d}.jpg")
            try:
                target.hardlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, target)
            new_row = {**source_row, "source_kind": source_row["source_kind"] + "_balanced_repeat", "crop": str(target)}
            rows.append(new_row)
            repeat += 1
            current_count += 1

    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter((row["split"], row["species"]) for row in rows)
    summary = {
        "status": "ready",
        "policy": "Ho Man Tin submitted reviews only; global route-block split; deleted proposals are hard negatives",
        "submitted_frames": len(submitted_frames),
        "positive_instances_before_minimum": sum(positive_counts.values()),
        "deleted_hard_negative_instances": len(negatives),
        "classes": len(class_names),
        "class_mapping": class_mapping,
        "ficus_policy": "Ficus microcarpa and Ficus benjamina merged to 榕树",
        "validation_blocks": sorted(val_blocks),
        "train_crops": sum(value for (split, _), value in split_counts.items() if split == "train"),
        "val_crops": sum(value for (split, _), value in split_counts.items() if split == "val"),
        "per_class": {
            species: {
                "source_positive_instances": positive_counts.get(species, 0),
                "train_crops": split_counts[("train", species)],
                "val_crops": split_counts[("val", species)],
            }
            for species in class_names
        },
        "balanced_train_target_per_class": target_per_class,
        "seed": cfg.seed,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
