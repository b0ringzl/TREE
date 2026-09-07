#!/usr/bin/env python3
"""Build leakage-controlled manifests for full-image pretraining and crop fine-tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PHOTO_PATTERN = re.compile(r"photo[_-](\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    path: Path
    source: str
    source_species: str
    species: str
    photo_id: str
    observation_id: str
    sha256: str
    annotated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-roots",
        type=Path,
        nargs="+",
        default=[
            PROJECT_ROOT / "external_datasets" / "inat_max150",
            PROJECT_ROOT / "external_datasets" / "inat_parent_name_supplement",
            PROJECT_ROOT / "external_datasets" / "inaturalist_dated" / "2026-06-07",
        ],
    )
    parser.add_argument(
        "--merge-map",
        type=Path,
        default=PROJECT_ROOT / "tools" / "tree_species_label_merge_map.csv",
    )
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-class-images", type=int, default=20)
    parser.add_argument("--minimum-class-groups", type=int, default=5)
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--minimum-crop-pixels", type=int, default=24)
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_merge_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            row["original_species"].strip(): row["label_species"].strip()
            for row in csv.DictReader(stream)
            if row.get("original_species") and row.get("label_species")
        }


def metadata_for_species(species_dir: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    metadata_path = species_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        return result
    for raw in metadata_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        local_path = str(row.get("local_path", ""))
        photo_id = str(row.get("photo_id", "") or "")
        observation_id = str(row.get("inat_observation_id", "") or "")
        if local_path:
            result[Path(local_path).name.casefold()] = (photo_id, observation_id)
        if photo_id:
            result[f"photo:{photo_id}"] = (photo_id, observation_id)
    return result


def source_name(root: Path) -> str:
    return relative(root).replace("/", ":")


def collect_candidates(roots: list[Path], merge_map: dict[str, str]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        species_dirs = sorted(
            {path.parent for path in root.rglob("images") if path.is_dir()},
            key=lambda path: str(path).casefold(),
        )
        for species_dir in species_dirs:
            if species_dir.name == "duplicate_photo_review":
                continue
            source_species = species_dir.name
            species = merge_map.get(source_species, source_species)
            metadata = metadata_for_species(species_dir)
            image_dir = species_dir / "images"
            for image_path in sorted(image_dir.rglob("*"), key=lambda path: str(path).casefold()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                match = PHOTO_PATTERN.search(image_path.stem)
                inferred_photo = match.group(1) if match else ""
                photo_id, observation_id = metadata.get(
                    image_path.name.casefold(), metadata.get(f"photo:{inferred_photo}", (inferred_photo, ""))
                )
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as error:
                    print(f"skip_unreadable={image_path}: {error}", flush=True)
                    continue
                candidates.append(
                    Candidate(
                        path=image_path.resolve(),
                        source=source_name(root),
                        source_species=source_species,
                        species=species,
                        photo_id=photo_id,
                        observation_id=observation_id,
                        sha256=sha256_file(image_path),
                        annotated=image_path.with_suffix(".txt").is_file(),
                    )
                )
    return candidates


def deduplicate(
    candidates: list[Candidate],
) -> tuple[list[Candidate], dict[str, int], list[dict[str, str]]]:
    # Roots are ordered by preference; annotated copies take precedence within a duplicate set.
    chosen: list[Candidate] = []
    photo_index: dict[str, int] = {}
    hash_index: dict[str, int] = {}
    quarantined: set[int] = set()
    reasons: Counter[str] = Counter()
    conflicts: list[dict[str, str]] = []
    for candidate in candidates:
        indices = []
        if candidate.photo_id and candidate.photo_id in photo_index:
            indices.append(photo_index[candidate.photo_id])
        if candidate.sha256 in hash_index:
            indices.append(hash_index[candidate.sha256])
        if indices:
            index = indices[0]
            existing = chosen[index]
            if existing.species != candidate.species:
                quarantined.add(index)
                reasons["conflicting_label_duplicates_quarantined"] += 1
                conflicts.append(
                    {
                        "photo_id": candidate.photo_id or existing.photo_id,
                        "sha256": candidate.sha256,
                        "first_species": existing.species,
                        "first_path": relative(existing.path),
                        "second_species": candidate.species,
                        "second_path": relative(candidate.path),
                    }
                )
                print(
                    f"quarantine_label_conflict={existing.path} ({existing.species}) <> "
                    f"{candidate.path} ({candidate.species})",
                    flush=True,
                )
                if candidate.photo_id:
                    photo_index[candidate.photo_id] = index
                hash_index[candidate.sha256] = index
                continue
            reasons["duplicate_photo_or_content"] += 1
            if candidate.annotated and not existing.annotated:
                chosen[index] = candidate
            if candidate.photo_id:
                photo_index[candidate.photo_id] = index
            hash_index[candidate.sha256] = index
            continue
        index = len(chosen)
        chosen.append(candidate)
        if candidate.photo_id:
            photo_index[candidate.photo_id] = index
        hash_index[candidate.sha256] = index
    return (
        [item for index, item in enumerate(chosen) if index not in quarantined],
        dict(reasons),
        conflicts,
    )


def group_id(item: Candidate) -> str:
    identity = item.observation_id or item.photo_id or item.sha256
    kind = "observation" if item.observation_id else "photo" if item.photo_id else "sha256"
    return f"{item.species}|{kind}:{identity}"


def select_classes(items: list[Candidate], minimum_images: int, minimum_groups: int) -> list[str]:
    images = Counter(item.species for item in items)
    groups: defaultdict[str, set[str]] = defaultdict(set)
    for item in items:
        groups[item.species].add(group_id(item))
    return sorted(
        [
            species
            for species, count in images.items()
            if count >= minimum_images and len(groups[species]) >= minimum_groups
        ],
        key=str.casefold,
    )


def allocate_splits(
    items: list[Candidate], classes: list[str], seed: int, val_fraction: float, test_fraction: float
) -> dict[str, str]:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 0.5:
        raise ValueError("Validation/test fractions must be positive and sum to less than 0.5")
    by_species: defaultdict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.species in classes:
            by_species[item.species].add(group_id(item))
    assignments: dict[str, str] = {}
    for species in classes:
        groups = sorted(
            by_species[species], key=lambda value: digest_text(f"{seed}|{species}|{value}")
        )
        test_count = max(1, round(len(groups) * test_fraction))
        val_count = max(1, round(len(groups) * val_fraction))
        if test_count + val_count >= len(groups):
            raise ValueError(f"Too few groups to split {species}: {len(groups)}")
        for index, value in enumerate(groups):
            assignments[value] = (
                "test" if index < test_count else "val" if index < test_count + val_count else "train"
            )
    return assignments


def parse_boxes(label_path: Path, padding: float) -> tuple[list[tuple[float, float, float, float]], int]:
    boxes: list[tuple[float, float, float, float]] = []
    invalid = 0
    for raw in label_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = raw.strip().split()
        try:
            values = [float(value) for value in parts]
        except ValueError:
            invalid += 1
            continue
        if len(values) == 5:
            _, cx, cy, width, height = values
            x0, y0, x1, y1 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        elif len(values) >= 7 and (len(values) - 1) % 2 == 0:
            coordinates = values[1:]
            xs, ys = coordinates[0::2], coordinates[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        else:
            invalid += 1
            continue
        if any(value < 0 or value > 1 for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            invalid += 1
            continue
        width, height = x1 - x0, y1 - y0
        boxes.append(
            (
                max(0.0, x0 - width * padding),
                max(0.0, y0 - height * padding),
                min(1.0, x1 + width * padding),
                min(1.0, y1 + height * padding),
            )
        )
    return boxes, invalid


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], classes: list[str]) -> dict[str, object]:
    split_counts = Counter(str(row["split"]) for row in rows)
    class_counts = {
        species: {
            split: sum(1 for row in rows if row["species"] == species and row["split"] == split)
            for split in ("train", "val", "test")
        }
        for species in classes
    }
    return {"classes": len(classes), "samples": len(rows), "splits": dict(split_counts), "per_class": class_counts}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_map = read_merge_map(args.merge_map.resolve())
    candidates = collect_candidates(args.image_roots, merge_map)
    unique, duplicate_summary, label_conflicts = deduplicate(candidates)
    full_classes = select_classes(unique, args.minimum_class_images, args.minimum_class_groups)
    assignments = allocate_splits(
        unique, full_classes, args.seed, args.val_fraction, args.test_fraction
    )
    full_class_to_idx = {name: index for index, name in enumerate(full_classes)}
    full_rows: list[dict[str, object]] = []
    selected_items: list[Candidate] = []
    for item in unique:
        if item.species not in full_class_to_idx:
            continue
        selected_items.append(item)
        identity = f"full|{relative(item.path)}|{item.sha256}"
        full_rows.append(
            {
                "sample_id": digest_text(identity)[:20],
                "split": assignments[group_id(item)],
                "source": item.source,
                "species": item.species,
                "source_species": item.source_species,
                "class_index": full_class_to_idx[item.species],
                "group_id": group_id(item),
                "image_path": relative(item.path),
                "sample_type": "full",
                "x0": "0.00000000",
                "y0": "0.00000000",
                "x1": "1.00000000",
                "y1": "1.00000000",
                "photo_id": item.photo_id,
                "observation_id": item.observation_id,
                "sha256": item.sha256,
            }
        )
    full_rows.sort(key=lambda row: (str(row["split"]), int(row["class_index"]), str(row["sample_id"])))

    annotated_items: list[tuple[Candidate, list[tuple[float, float, float, float]]]] = []
    crop_skipped: Counter[str] = Counter()
    for item in selected_items:
        label_path = item.path.with_suffix(".txt")
        if not label_path.is_file():
            continue
        boxes, invalid = parse_boxes(label_path, args.crop_padding)
        crop_skipped["invalid_annotation_rows"] += invalid
        if boxes:
            annotated_items.append((item, boxes))
    crop_species = sorted({item.species for item, _ in annotated_items}, key=str.casefold)
    crop_class_to_idx = {name: index for index, name in enumerate(crop_species)}
    crop_rows: list[dict[str, object]] = []
    for item, boxes in annotated_items:
        with Image.open(item.path) as image:
            image_width, image_height = image.size
        for annotation_index, (x0, y0, x1, y1) in enumerate(boxes):
            if (x1 - x0) * image_width < args.minimum_crop_pixels or (y1 - y0) * image_height < args.minimum_crop_pixels:
                crop_skipped["crop_too_small"] += 1
                continue
            identity = f"crop|{relative(item.path)}|{annotation_index}|{item.species}"
            crop_rows.append(
                {
                    "sample_id": digest_text(identity)[:20],
                    "split": assignments[group_id(item)],
                    "source": item.source,
                    "species": item.species,
                    "source_species": item.source_species,
                    "class_index": crop_class_to_idx[item.species],
                    "group_id": group_id(item),
                    "image_path": relative(item.path),
                    "sample_type": "crop",
                    "x0": f"{x0:.8f}",
                    "y0": f"{y0:.8f}",
                    "x1": f"{x1:.8f}",
                    "y1": f"{y1:.8f}",
                    "photo_id": item.photo_id,
                    "observation_id": item.observation_id,
                    "sha256": item.sha256,
                }
            )
    crop_rows.sort(key=lambda row: (str(row["split"]), int(row["class_index"]), str(row["sample_id"])))

    if label_conflicts:
        write_csv(output_dir / "label_conflicts.csv", label_conflicts)

    for rows, classes, name in (
        (full_rows, full_classes, "full_manifest.csv"),
        (crop_rows, crop_species, "crop_manifest.csv"),
    ):
        for split in ("train", "val", "test"):
            present = {row["species"] for row in rows if row["split"] == split}
            missing = set(classes) - present
            if missing:
                raise ValueError(f"{name} {split} is missing classes: {sorted(missing)}")
        write_csv(output_dir / name, rows)

    hash_splits: defaultdict[str, set[str]] = defaultdict(set)
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in full_rows + crop_rows:
        hash_splits[str(row["sha256"])].add(str(row["split"]))
        group_splits[str(row["group_id"])].add(str(row["split"]))
    leaking_hashes = [key for key, splits in hash_splits.items() if len(splits) > 1]
    leaking_groups = [key for key, splits in group_splits.items() if len(splits) > 1]
    if leaking_hashes or leaking_groups:
        raise ValueError(f"Leakage detected: hashes={len(leaking_hashes)}, groups={len(leaking_groups)}")

    summary = {
        "schema_version": 1,
        "status": "validated",
        "seed": args.seed,
        "roots": [relative(path.resolve()) for path in args.image_roots],
        "merge_map": relative(args.merge_map.resolve()),
        "raw_candidates": len(candidates),
        "unique_images": len(unique),
        "deduplication": duplicate_summary,
        "label_conflicts": {
            "count": len(label_conflicts),
            "manifest": "label_conflicts.csv" if label_conflicts else None,
        },
        "thresholds": {
            "minimum_class_images": args.minimum_class_images,
            "minimum_class_groups": args.minimum_class_groups,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "crop_padding": args.crop_padding,
        },
        "full": summarize(full_rows, full_classes),
        "crop": summarize(crop_rows, crop_species),
        "crop_skipped": dict(crop_skipped),
        "class_names": {"full": full_classes, "crop": crop_species},
        "checks": {
            "observation_group_leakage": False,
            "exact_content_leakage": False,
            "all_classes_in_every_split": True,
            "crop_split_inherited_from_full_image": True,
        },
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
