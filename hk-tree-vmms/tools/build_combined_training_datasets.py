from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


DATASETS = {
    "jianshazui": "jianshazui_frame_labeler",
    "hewentian": "hewentian_frame_labeler",
    "stubbs_road": "stubbs_road_frame_labeler",
}
COMMON_CLASSES = (
    ("Ficus_microcarpa", "Ficus microcarpa 榕樹(細葉榕)"),
    ("Livistona_chinensis", "Livistona chinensis 蒲葵"),
    ("Wodyetia_bifurcata", "Wodyetia bifurcata 狐尾椰子"),
)
SPLITS = ("train", "val", "test")


def latest_export(project_root: Path, labeler_name: str) -> Path:
    root = project_root / "derived" / labeler_name / "training_exports"
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("effective_yolo_")
        and "superseded" not in path.name
        and (path / "export_summary.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No effective export found under {root}")
    return candidates[-1]


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records(exports: dict[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset, export in exports.items():
        for split in SPLITS:
            for path in sorted((export / "records" / split).glob("*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                records.append(
                    {
                        **record,
                        "source_dataset": dataset,
                        "source_export": str(export),
                        "source_split": split,
                        "source_record": str(path),
                        "combined_block_id": f"{dataset}__{record['route_block_id']}",
                    }
                )
    return records


def build_tree_segmentation(
    records: list[dict[str, Any]], exports: dict[str, Path], output: Path
) -> dict[str, Any]:
    manifest: list[dict[str, Any]] = []
    split_frames = Counter()
    split_instances = Counter()
    source_frames = Counter()
    source_instances = Counter()
    link_modes = Counter()

    for record in records:
        split = record["source_split"]
        dataset = record["source_dataset"]
        stem = f"{dataset}__{record['frame_key']}"
        source_image = Path(record["linked_image"])
        image_target = output / "images" / split / f"{stem}.jpg"
        label_target = output / "labels" / split / f"{stem}.txt"
        record_target = output / "records" / split / f"{stem}.json"
        link_modes[link_or_copy(source_image, image_target)] += 1

        lines: list[str] = []
        labels: list[dict[str, Any]] = []
        for label in record["labels"]:
            values = ["0"]
            for x, y in label["points"]:
                values.extend((f"{float(x):.8f}", f"{float(y):.8f}"))
            lines.append(" ".join(values))
            labels.append({**label, "segmentation_class_id": 0})
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(
            record_target,
            {
                **record,
                "combined_image": str(image_target),
                "combined_label": str(label_target),
                "labels": labels,
            },
        )

        split_frames[split] += 1
        split_instances[split] += len(labels)
        source_frames[dataset] += 1
        source_instances[dataset] += len(labels)
        manifest.append(
            {
                "split": split,
                "source_dataset": dataset,
                "stream_id": record["stream_id"],
                "frame_key": record["frame_key"],
                "route_block_id": record["combined_block_id"],
                "image": str(image_target),
                "label": str(label_target),
                "tree_count": len(labels),
                "source_record": record["source_record"],
                "exposure_ev": (record.get("preprocess") or {}).get("exposure_ev", 0.0),
                "contrast": (record.get("preprocess") or {}).get("contrast", 1.0),
            }
        )

    write_csv(output / "manifest.csv", manifest)
    (output / "data.yaml").write_text(
        f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: tree\n",
        encoding="utf-8",
    )
    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "class-agnostic street-tree instance segmentation",
        "source_exports": {key: str(value) for key, value in exports.items()},
        "images": len(records),
        "instances": sum(len(record["labels"]) for record in records),
        "classes": ["tree"],
        "split_frames": dict(split_frames),
        "split_instances": dict(split_instances),
        "source_frames": dict(source_frames),
        "source_instances": dict(source_instances),
        "link_modes": dict(link_modes),
        "split_policy": "reuse each source export's leakage-safe 100 m route-block split",
    }
    write_json(output / "export_summary.json", summary)
    return summary


def photo_lookup(exposure_ev: float, contrast: float) -> list[int]:
    exposure_factor = 2.0 ** float(exposure_ev)
    contrast_factor = float(contrast)
    return [
        max(
            0,
            min(
                255,
                int(round(((value * exposure_factor) - 127.5) * contrast_factor + 127.5)),
            ),
        )
        for value in range(256)
    ]


def square_crop_box(
    points: list[list[float]], width: int, height: int, padding: float = 0.12
) -> tuple[int, int, int, int, int, int]:
    xs = [float(point[0]) * width for point in points]
    ys = [float(point[1]) * height for point in points]
    polygon_width = max(xs) - min(xs)
    polygon_height = max(ys) - min(ys)
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    side = max(32.0, max(polygon_width, polygon_height) * (1.0 + 2.0 * padding))
    side = min(side, float(min(width, height)))
    left = center_x - side / 2
    top = center_y - side / 2
    left = min(max(0.0, left), width - side)
    top = min(max(0.0, top), height - side)
    right = left + side
    bottom = top + side
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(right)),
        int(math.ceil(bottom)),
        int(round(polygon_width)),
        int(round(polygon_height)),
    )


def create_common_instances(
    records: list[dict[str, Any]], output: Path
) -> list[dict[str, Any]]:
    species_to_directory = {species: directory for directory, species in COMMON_CLASSES}
    instances: list[dict[str, Any]] = []
    for record in records:
        common_labels = [
            label for label in record["labels"] if label["species"] in species_to_directory
        ]
        if not common_labels:
            continue
        image = Image.open(record["linked_image"]).convert("RGB")
        preprocess = record.get("preprocess") or {}
        exposure_ev = float(preprocess.get("exposure_ev", 0.0) or 0.0)
        contrast = float(preprocess.get("contrast", 1.0) or 1.0)
        changed = abs(exposure_ev) >= 0.01 or abs(contrast - 1.0) >= 0.01
        lookup = photo_lookup(exposure_ev, contrast) if changed else None
        for ordinal, label in enumerate(record["labels"], start=1):
            species = label["species"]
            if species not in species_to_directory:
                continue
            directory = species_to_directory[species]
            box = square_crop_box(label["points"], *image.size)
            left, top, right, bottom, polygon_width, polygon_height = box
            crop = image.crop((left, top, right, bottom))
            stem = f"{record['source_dataset']}__{record['frame_key']}__tree_{ordinal:02d}"
            raw_path = output / "instances" / "raw" / directory / f"{stem}.jpg"
            adjusted_path = output / "instances" / "adjusted" / directory / f"{stem}.jpg"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(raw_path, format="JPEG", quality=95, subsampling=0)
            if lookup is None:
                link_or_copy(raw_path, adjusted_path)
            else:
                adjusted_path.parent.mkdir(parents=True, exist_ok=True)
                crop.point(lookup * 3).save(
                    adjusted_path, format="JPEG", quality=95, subsampling=0
                )
            instances.append(
                {
                    "instance_id": stem,
                    "source_dataset": record["source_dataset"],
                    "stream_id": record["stream_id"],
                    "frame_key": record["frame_key"],
                    "route_block_id": record["combined_block_id"],
                    "species": species,
                    "class_directory": directory,
                    "raw_crop": str(raw_path),
                    "adjusted_crop": str(adjusted_path),
                    "source_image": record["linked_image"],
                    "source_record": record["source_record"],
                    "polygon_width_px": polygon_width,
                    "polygon_height_px": polygon_height,
                    "crop_width_px": right - left,
                    "crop_height_px": bottom - top,
                    "low_resolution": min(polygon_width, polygon_height) < 64,
                    "exposure_ev": exposure_ev,
                    "contrast": contrast,
                    "preprocess_changed": changed,
                }
            )
    return instances


def choose_validation_blocks(
    candidates: list[dict[str, Any]], seed: int
) -> set[str]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in candidates:
        by_block[instance["route_block_id"]].append(instance)
    blocks = sorted(by_block)
    if len(blocks) < 3:
        raise ValueError("At least three training route blocks are required")
    total = Counter(instance["species"] for instance in candidates)
    expected = max(1, round(len(blocks) * 0.15))
    randomizer = random.Random(seed)
    best: tuple[float, set[str]] | None = None
    sizes = range(max(1, expected - 1), min(len(blocks) - 1, expected + 3) + 1)
    for size in sizes:
        for _ in range(10_000):
            selected = set(randomizer.sample(blocks, size))
            val = Counter(
                instance["species"]
                for block in selected
                for instance in by_block[block]
            )
            train = total - val
            if any(val[species] == 0 or train[species] == 0 for _, species in COMMON_CLASSES):
                continue
            val_count = sum(val.values())
            ratio_error = abs(val_count / sum(total.values()) - 0.15)
            class_error = sum(
                abs(val[species] / total[species] - 0.15)
                for _, species in COMMON_CLASSES
            ) / len(COMMON_CLASSES)
            route_presence = {
                instance["source_dataset"]
                for block in selected
                for instance in by_block[block]
            }
            route_penalty = max(
                0,
                len({item["source_dataset"] for item in candidates})
                - len(route_presence),
            )
            score = 3.0 * ratio_error + class_error + route_penalty
            if best is None or score < best[0]:
                best = score, selected
    if best is None:
        raise RuntimeError("Could not create a class-complete validation block split")
    return best[1]


def build_common_folds(
    instances: list[dict[str, Any]], exports: dict[str, Path], output: Path, seed: int
) -> dict[str, Any]:
    class_map = [
        {"class_id": index, "directory": directory, "species": species}
        for index, (directory, species) in enumerate(COMMON_CLASSES)
    ]
    write_json(output / "class_map.json", class_map)
    write_csv(output / "instances.csv", instances)
    folds: dict[str, Any] = {}

    for fold_number, holdout in enumerate(DATASETS, start=1):
        fold_name = f"holdout_{holdout}"
        candidates = [item for item in instances if item["source_dataset"] != holdout]
        validation_blocks = choose_validation_blocks(candidates, seed + fold_number)
        fold_rows: list[dict[str, Any]] = []
        split_counts: dict[str, Counter[str]] = defaultdict(Counter)
        split_routes: dict[str, Counter[str]] = defaultdict(Counter)
        split_blocks: dict[str, set[str]] = defaultdict(set)
        link_modes = Counter()
        for instance in instances:
            if instance["source_dataset"] == holdout:
                split = "test"
            elif instance["route_block_id"] in validation_blocks:
                split = "val"
            else:
                split = "train"
            for variant in ("raw", "adjusted"):
                source = Path(instance[f"{variant}_crop"])
                target = (
                    output
                    / "folds"
                    / fold_name
                    / variant
                    / split
                    / instance["class_directory"]
                    / source.name
                )
                link_modes[link_or_copy(source, target)] += 1
            split_counts[split][instance["species"]] += 1
            split_routes[split][instance["source_dataset"]] += 1
            split_blocks[split].add(instance["route_block_id"])
            fold_rows.append({"fold": fold_name, "split": split, **instance})
        write_csv(output / "folds" / fold_name / "manifest.csv", fold_rows)
        folds[fold_name] = {
            "holdout_route": holdout,
            "split_instances": {
                split: sum(counts.values()) for split, counts in split_counts.items()
            },
            "split_species": {
                split: dict(counts) for split, counts in split_counts.items()
            },
            "split_routes": {
                split: dict(counts) for split, counts in split_routes.items()
            },
            "split_blocks": {
                split: sorted(blocks) for split, blocks in split_blocks.items()
            },
            "link_modes": dict(link_modes),
        }

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "three-species single-tree classification with leave-one-route-out evaluation",
        "source_exports": {key: str(value) for key, value in exports.items()},
        "instances": len(instances),
        "species_counts": dict(Counter(item["species"] for item in instances)),
        "source_counts": dict(Counter(item["source_dataset"] for item in instances)),
        "preprocess_changed_instances": sum(item["preprocess_changed"] for item in instances),
        "low_resolution_instances": sum(item["low_resolution"] for item in instances),
        "variants": ["raw", "adjusted"],
        "class_map": class_map,
        "folds": folds,
        "split_policy": "test is one complete route; validation uses whole 100 m blocks from the other routes",
    }
    write_json(output / "export_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined tree segmentation and common-three classification datasets."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    exports = {
        dataset: latest_export(project_root, labeler)
        for dataset, labeler in DATASETS.items()
    }
    records = load_records(exports)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    segmentation_output = (
        project_root
        / "derived"
        / "combined_tree_segmentation"
        / f"effective_yolo_{timestamp}"
    )
    classification_output = (
        project_root
        / "derived"
        / "common3_species_crops"
        / f"effective_{timestamp}"
    )
    segmentation_summary = build_tree_segmentation(
        records, exports, segmentation_output
    )
    instances = create_common_instances(records, classification_output)
    classification_summary = build_common_folds(
        instances, exports, classification_output, args.seed
    )
    print(
        json.dumps(
            {
                "segmentation_output": str(segmentation_output),
                "segmentation_summary": segmentation_summary,
                "classification_output": str(classification_output),
                "classification_summary": classification_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
