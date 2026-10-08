#!/usr/bin/env python3
"""Build a leakage-controlled ten-class image-crop manifest.

The two source datasets remain read-only. YOLO boxes and polygons are converted
to normalized bounding boxes in a CSV manifest; crops are created in memory by
the training loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = (
    "Acacia auriculiformis",
    "Acacia confusa",
    "Acacia mangium",
    "Adenanthera microsperma",
    "Adinandra millettii",
    "Ailanthus fordii",
    "Albizia lebbeck",
    "Araucaria columnaris(Araucaria heterophylla)",
    "Areca catechu",
    "Bauhinia purpurea",
)
WEB_ALIASES = {
    "Araucaria columnaris": "Araucaria columnaris(Araucaria heterophylla)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--web-root",
        type=Path,
        default=PROJECT_ROOT / "external_datasets" / "inat_max150",
    )
    parser.add_argument(
        "--hk-root", type=Path, default=PROJECT_ROOT / "dataset"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--minimum-pixels", type=int, default=24)
    return parser.parse_args()


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_annotations(path: Path) -> list[tuple[str, float, float, float, float]]:
    if not path.is_file():
        return []
    annotations: list[tuple[str, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = line.strip().split()
        try:
            values = [float(value) for value in parts]
        except ValueError:
            continue
        if len(values) == 5:
            _, center_x, center_y, width, height = values
            if not (
                0 <= center_x <= 1
                and 0 <= center_y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                continue
            x0 = center_x - width / 2
            y0 = center_y - height / 2
            x1 = center_x + width / 2
            y1 = center_y + height / 2
            kind = "box"
        elif len(values) >= 7 and (len(values) - 1) % 2 == 0:
            coordinates = values[1:]
            if any(value < 0 or value > 1 for value in coordinates):
                continue
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            kind = "polygon"
        else:
            continue
        if x1 > x0 and y1 > y0:
            annotations.append((kind, x0, y0, x1, y1))
    return annotations


def padded_box(
    box: tuple[float, float, float, float], padding: float
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    return (
        max(0.0, x0 - width * padding),
        max(0.0, y0 - height * padding),
        min(1.0, x1 + width * padding),
        min(1.0, y1 + height * padding),
    )


def image_records(
    image_path: Path,
    *,
    source: str,
    species: str,
    group_id: str,
    split: str,
    padding: float,
    minimum_pixels: int,
) -> tuple[list[dict[str, object]], Counter[str]]:
    label_path = image_path.with_suffix(".txt")
    parsed = parse_annotations(label_path)
    reasons: Counter[str] = Counter()
    if not parsed:
        reasons["missing_or_invalid_annotation"] += 1
        return [], reasons
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
    except Exception:
        reasons["unreadable_image"] += 1
        return [], reasons

    rows: list[dict[str, object]] = []
    for annotation_index, (kind, x0, y0, x1, y1) in enumerate(parsed):
        x0, y0, x1, y1 = padded_box((x0, y0, x1, y1), padding)
        if (x1 - x0) * width < minimum_pixels or (y1 - y0) * height < minimum_pixels:
            reasons["crop_too_small"] += 1
            continue
        identity = f"{source}|{relative(image_path)}|{annotation_index}|{species}"
        rows.append(
            {
                "sample_id": stable_digest(identity)[:20],
                "split": split,
                "source": source,
                "species": species,
                "class_index": CLASS_NAMES.index(species),
                "group_id": group_id,
                "image_path": relative(image_path),
                "label_path": relative(label_path),
                "annotation_index": annotation_index,
                "annotation_type": kind,
                "x0": f"{x0:.8f}",
                "y0": f"{y0:.8f}",
                "x1": f"{x1:.8f}",
                "y1": f"{y1:.8f}",
                "image_width": width,
                "image_height": height,
            }
        )
    return rows, reasons


def allocate_hk_splits(
    hk_root: Path, seed: int, fraction: float
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
    dict[str, dict[str, int]],
]:
    if not 0 < fraction < 0.5:
        raise ValueError("holdout fraction must be between 0 and 0.5")
    assignments: dict[tuple[str, str], str] = {}
    component_ids: dict[tuple[str, str], str] = {}
    summary: dict[str, dict[str, int]] = {}
    for species in CLASS_NAMES:
        species_root = hk_root / species
        annotated_images = [
            image
            for image in sorted(species_root.rglob("*"))
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
            and parse_annotations(image.with_suffix(".txt"))
        ]
        groups = sorted(
            {image.parent.relative_to(species_root).as_posix() for image in annotated_images}
        )
        parent = {group: group for group in groups}

        def find(group: str) -> str:
            while parent[group] != group:
                parent[group] = parent[parent[group]]
                group = parent[group]
            return group

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            keep, merge = sorted((left_root, right_root))
            parent[merge] = keep

        hash_groups: defaultdict[str, set[str]] = defaultdict(set)
        for image in annotated_images:
            group = image.parent.relative_to(species_root).as_posix()
            hash_groups[sha256_file(image)].add(group)
        for duplicate_groups in hash_groups.values():
            ordered = sorted(duplicate_groups)
            for other in ordered[1:]:
                union(ordered[0], other)

        components: defaultdict[str, list[str]] = defaultdict(list)
        for group in groups:
            components[find(group)].append(group)
        ordered_components = sorted(
            (tuple(sorted(members)) for members in components.values()),
            key=lambda members: stable_digest(
                f"{seed}|{species}|{'|'.join(members)}"
            ),
        )
        if len(ordered_components) < 2:
            raise ValueError(f"{species} has fewer than two labeled HK tree groups")
        test_count = max(1, round(len(ordered_components) * fraction))
        val_count = max(1, round(len(ordered_components) * fraction))
        while test_count + val_count > len(ordered_components):
            if val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        for index, members in enumerate(ordered_components):
            split = (
                "test"
                if index < test_count
                else "val"
                if index < test_count + val_count
                else "train"
            )
            component_id = stable_digest(f"{species}|{'|'.join(members)}")[:16]
            for group in members:
                assignments[(species, group)] = split
                component_ids[(species, group)] = component_id
        split_counts = Counter(
            assignments[(species, members[0])] for members in ordered_components
        )
        summary[species] = {
            "total": len(ordered_components),
            "raw_tree_folders": len(groups),
            "duplicate_linked_components": sum(
                1 for members in ordered_components if len(members) > 1
            ),
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
        }
    return assignments, component_ids, summary


def main() -> None:
    args = parse_args()
    web_root = args.web_root.resolve()
    hk_root = args.hk_root.resolve()
    output_dir = args.output_dir.resolve()
    if not web_root.is_dir() or not hk_root.is_dir():
        raise FileNotFoundError("Both source dataset roots must exist")
    output_dir.mkdir(parents=True, exist_ok=True)

    hk_assignments, hk_component_ids, hk_group_summary = allocate_hk_splits(
        hk_root, args.seed, args.holdout_fraction
    )
    records: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()
    seen_web_hashes: set[tuple[str, str]] = set()
    seen_hk_hashes: set[tuple[str, str]] = set()

    web_directories: dict[str, Path] = {}
    for directory in web_root.iterdir():
        if not directory.is_dir():
            continue
        canonical = WEB_ALIASES.get(directory.name, directory.name)
        if canonical in CLASS_NAMES and canonical not in web_directories:
            web_directories[canonical] = directory
    if set(web_directories) != set(CLASS_NAMES):
        missing = sorted(set(CLASS_NAMES) - set(web_directories))
        raise ValueError(f"Missing web species directories: {missing}")

    for species in CLASS_NAMES:
        for image_path in sorted(web_directories[species].rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_hash = sha256_file(image_path)
            web_hash_key = (species, image_hash)
            if web_hash_key in seen_web_hashes:
                skipped["duplicate_web_image"] += 1
                continue
            seen_web_hashes.add(web_hash_key)
            rows, reasons = image_records(
                image_path,
                source="web",
                species=species,
                group_id=f"web:{species}:{relative(image_path)}",
                split="train",
                padding=args.padding,
                minimum_pixels=args.minimum_pixels,
            )
            records.extend(rows)
            skipped.update(reasons)

        species_root = hk_root / species
        for image_path in sorted(species_root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            group = image_path.parent.relative_to(species_root).as_posix()
            split = hk_assignments.get((species, group))
            if split is None:
                continue
            image_hash = sha256_file(image_path)
            hk_hash_key = (species, image_hash)
            if hk_hash_key in seen_hk_hashes:
                skipped["duplicate_hk_image"] += 1
                continue
            seen_hk_hashes.add(hk_hash_key)
            rows, reasons = image_records(
                image_path,
                source="hk_streetview",
                species=species,
                group_id=f"hk:{species}:{hk_component_ids[(species, group)]}",
                split=split,
                padding=args.padding,
                minimum_pixels=args.minimum_pixels,
            )
            records.extend(rows)
            skipped.update(reasons)

    records.sort(
        key=lambda row: (
            str(row["split"]),
            int(row["class_index"]),
            str(row["group_id"]),
            str(row["sample_id"]),
        )
    )
    sample_ids = [str(row["sample_id"]) for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs detected")

    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["source"] == "hk_streetview":
            group_splits[str(row["group_id"])].add(str(row["split"]))
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"HK tree groups cross splits: {leaked}")

    content_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in records:
        image_hash = sha256_file(PROJECT_ROOT / str(row["image_path"]))
        content_splits[image_hash].add(str(row["split"]))
    content_leaks = {
        image_hash: splits
        for image_hash, splits in content_splits.items()
        if len(splits) > 1
    }
    if content_leaks:
        raise ValueError(f"Exact image content crosses splits: {content_leaks}")

    for split in ("train", "val", "test"):
        present = {str(row["species"]) for row in records if row["split"] == split}
        if present != set(CLASS_NAMES):
            raise ValueError(f"{split} does not contain all ten classes: {sorted(present)}")
    if any(row["source"] != "hk_streetview" for row in records if row["split"] != "train"):
        raise ValueError("Validation and test must contain only HK Street View samples")

    fields = list(records[0])
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    split_counts = Counter(str(row["split"]) for row in records)
    source_counts = Counter(str(row["source"]) for row in records)
    class_split_counts: dict[str, dict[str, int]] = {}
    for species in CLASS_NAMES:
        class_split_counts[species] = {
            split: sum(
                1
                for row in records
                if row["species"] == species and row["split"] == split
            )
            for split in ("train", "val", "test")
        }
    summary = {
        "schema_version": 1,
        "stage": "independent-hk-web-image-resnet50-ten-class",
        "status": "validated",
        "seed": args.seed,
        "class_names": list(CLASS_NAMES),
        "sources": {
            "web_root": relative(web_root),
            "hk_root": relative(hk_root),
        },
        "policy": {
            "web": "training only",
            "hk_streetview": "tree-folder grouped train/val/test",
            "validation_and_test_domain": "HK Street View only",
            "holdout_fraction_per_split": args.holdout_fraction,
            "bbox_padding": args.padding,
            "minimum_crop_pixels": args.minimum_pixels,
        },
        "sample_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "class_split_counts": class_split_counts,
        "hk_group_counts": hk_group_summary,
        "skipped": dict(sorted(skipped.items())),
        "checks": {
            "ten_classes_in_every_split": True,
            "hk_group_leakage": False,
            "exact_content_leakage": False,
            "non_hk_holdout_samples": 0,
            "unique_sample_ids": True,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "limitations": [
            "Bauhinia purpurea has only two labeled HK tree groups, allocated one to validation and one to test.",
            "Web and HK annotations mix YOLO boxes and polygons; polygons are converted to padded bounding boxes.",
        ],
    }
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
