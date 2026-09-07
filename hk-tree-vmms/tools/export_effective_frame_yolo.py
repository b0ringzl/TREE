from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SPECIES_ALIASES = {
    "Archontophoenix alexandrae 假檳榔": "Wodyetia bifurcata 狐尾椰子",
    "Archontophoenix alexandrae 假槟榔": "Wodyetia bifurcata 狐尾椰子",
    "Araucaria heterophylla异叶南阳杉": "Araucaria heterophylla 异叶南洋杉",
    "Araucaria heterophylla 异叶南阳杉": "Araucaria heterophylla 异叶南洋杉",
    "Araucaria heterophylla异叶南洋杉": "Araucaria heterophylla 异叶南洋杉",
    "Alstonia scholaris": "Alstonia scholaris 糖膠樹",
    "Lagerstroemia indica": "Lagerstroemia indica 紫薇",
    "紫薇": "Lagerstroemia indica 紫薇",
    "Ravenala madagascariensis": "Ravenala madagascariensis 旅人蕉",
    "旅人蕉": "Ravenala madagascariensis 旅人蕉",
}
DEFAULT_STREAM_IDS = ("jianshazui_pano_1", "jianshazui_pano_2")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class FrameInfo:
    frame_key: str
    frame_id: str
    stream_id: str
    seq_id: int
    route_distance_m: float
    source_image: Path
    hong_kong_datetime: str
    longitude: float
    latitude: float

    @property
    def route_block_id(self) -> str:
        return f"{self.stream_id}_{int(self.route_distance_m // 100):04d}"


def normalize_species(value: str) -> str:
    stripped = str(value).strip()
    return SPECIES_ALIASES.get(stripped, stripped)


def orientation(a: list[float], b: list[float], c: list[float]) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if value > 1e-12:
        return 1
    if value < -1e-12:
        return -1
    return 0


def has_proper_self_intersection(points: list[list[float]]) -> bool:
    count = len(points)
    for first in range(count):
        for second in range(first + 1, count):
            if second == (first + 1) % count or (second + 1) % count == first:
                continue
            a, b = points[first], points[(first + 1) % count]
            c, d = points[second], points[(second + 1) % count]
            if (
                orientation(a, b, c) * orientation(a, b, d) < 0
                and orientation(c, d, a) * orientation(c, d, b) < 0
            ):
                return True
    return False


def load_frames(
    coordinate_csv: Path, vmms_root: Path, stream_ids: tuple[str, ...]
) -> dict[str, FrameInfo]:
    grouped: dict[str, list[dict[str, str]]] = {stream: [] for stream in stream_ids}
    with coordinate_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] in grouped:
                grouped[row["stream_id"]].append(row)

    frames: dict[str, FrameInfo] = {}
    multi_stream = len(stream_ids) > 1
    for stream_id in stream_ids:
        route_distance = 0.0
        previous: tuple[float, float] | None = None
        for row in sorted(grouped[stream_id], key=lambda item: int(item["seq_id"])):
            current = (float(row["hk80_easting"]), float(row["hk80_northing"]))
            if previous is not None:
                route_distance += math.dist(previous, current)
            previous = current
            frame_id = row["frame_id"]
            frame_key = f"{stream_id}__{frame_id}" if multi_stream else frame_id
            frames[frame_key] = FrameInfo(
                frame_key=frame_key,
                frame_id=frame_id,
                stream_id=stream_id,
                seq_id=int(row["seq_id"]),
                route_distance_m=route_distance,
                source_image=(vmms_root / row["source_image_relpath"]).resolve(),
                hong_kong_datetime=row["hong_kong_datetime"],
                longitude=float(row["wgs84_longitude"]),
                latitude=float(row["wgs84_latitude"]),
            )
    return frames


def validate_labels(labels: list[dict[str, Any]], frame_key: str) -> None:
    for number, label in enumerate(labels, start=1):
        points = label.get("points") or []
        if len(points) < 3:
            raise ValueError(f"{frame_key} label {number} has fewer than 3 points")
        if any(
            len(point) != 2 or not 0 <= float(point[0]) <= 1 or not 0 <= float(point[1]) <= 1
            for point in points
        ):
            raise ValueError(f"{frame_key} label {number} has invalid normalized points")
        if has_proper_self_intersection(points):
            raise ValueError(f"{frame_key} label {number} is self-intersecting")


def choose_block_splits(
    records: list[dict[str, Any]], seed: int, stream_ids: tuple[str, ...]
) -> dict[str, str]:
    block_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    species_blocks: dict[str, set[str]] = defaultdict(set)
    for record in records:
        block = record["frame"].route_block_id
        block_records[block].append(record)
        for species in set(record["species"]):
            species_blocks[species].add(block)

    blocks = sorted(block_records)
    if len(blocks) < 3:
        raise ValueError("At least three labeled route blocks are required")
    locked_train = {
        block
        for species, occupied_blocks in species_blocks.items()
        if len(occupied_blocks) <= 2
        for block in occupied_blocks
    }
    target_block_counts = {
        "val": max(1, round(len(blocks) * SPLIT_RATIOS["val"])),
        "test": max(1, round(len(blocks) * SPLIT_RATIOS["test"])),
    }
    target_block_counts["train"] = len(blocks) - sum(target_block_counts.values())
    target_block_counts["train"] = max(
        target_block_counts["train"], len(locked_train)
    )
    overflow = sum(target_block_counts.values()) - len(blocks)
    while overflow > 0:
        candidate = max(("val", "test"), key=lambda split: target_block_counts[split])
        if target_block_counts[candidate] <= 1:
            break
        target_block_counts[candidate] -= 1
        overflow -= 1

    remaining = [block for block in blocks if block not in locked_train]
    val_count = min(target_block_counts["val"], max(1, len(remaining) // 2))
    test_count = min(target_block_counts["test"], max(1, len(remaining) - val_count))
    total_frames = len(records)
    total_species = Counter(
        species for record in records for species in record["species"]
    )
    randomizer = random.Random(seed)
    best_score = float("inf")
    best_assignment: dict[str, str] | None = None

    for _ in range(20_000):
        shuffled = remaining.copy()
        randomizer.shuffle(shuffled)
        assignment = {block: "train" for block in locked_train}
        for block in shuffled[:val_count]:
            assignment[block] = "val"
        for block in shuffled[val_count : val_count + test_count]:
            assignment[block] = "test"
        for block in shuffled[val_count + test_count :]:
            assignment[block] = "train"

        frame_counts = Counter()
        split_species: dict[str, Counter[str]] = defaultdict(Counter)
        stream_presence: dict[str, set[str]] = defaultdict(set)
        for record in records:
            split = assignment[record["frame"].route_block_id]
            frame_counts[split] += 1
            split_species[split].update(record["species"])
            stream_presence[split].add(record["frame"].stream_id)

        if any(frame_counts[split] == 0 for split in SPLIT_RATIOS):
            continue
        if any(split_species["train"][species] == 0 for species in total_species):
            continue
        frame_error = sum(
            abs(frame_counts[split] / total_frames - target)
            for split, target in SPLIT_RATIOS.items()
        )
        species_error = 0.0
        for species, total in total_species.items():
            if len(species_blocks[species]) < 3:
                continue
            species_error += sum(
                abs(split_species[split][species] / total - SPLIT_RATIOS[split])
                for split in SPLIT_RATIOS
            )
        species_error /= max(1, sum(len(value) >= 3 for value in species_blocks.values()))
        missing_stream_penalty = sum(
            len(stream_ids) - len(stream_presence[split]) for split in SPLIT_RATIOS
        )
        score = 4.0 * frame_error + species_error + missing_stream_penalty
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not construct a leakage-safe route-block split")
    return best_assignment


def link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all effective frame annotations to YOLO segmentation."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeler-name", default="jianshazui_frame_labeler")
    parser.add_argument(
        "--streams",
        default=",".join(DEFAULT_STREAM_IDS),
        help="Comma-separated stream IDs; single-stream frame keys remain bare frame IDs.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    stream_ids = tuple(value.strip() for value in args.streams.split(",") if value.strip())
    if not stream_ids:
        raise ValueError("At least one stream ID is required")
    base = project_root / "derived" / args.labeler_name
    coordinate_csv = project_root / "outputs" / "coordinates" / "frame_coordinates.csv"
    vmms_root = project_root.parent.parent / "vmms"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output.resolve()
        if args.output
        else base / "training_exports" / f"effective_yolo_{timestamp}"
    )
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    draft_path = base / "runtime" / "frame_drafts.json"
    review_path = base / "runtime" / "frame_review_state.json"
    drafts = (
        json.loads(draft_path.read_text(encoding="utf-8"))
        if draft_path.is_file()
        else {}
    )
    reviews = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.is_file()
        else {}
    )
    global_classes = json.loads(
        (base / "annotations" / "classes.json").read_text(encoding="utf-8")
    )
    frames = load_frames(coordinate_csv, vmms_root, stream_ids)

    effective_records: list[dict[str, Any]] = []
    for frame_key in sorted(set(drafts) | set(reviews)):
        current = drafts.get(frame_key) or reviews.get(frame_key) or {}
        labels = current.get("labels") or []
        if not labels:
            continue
        if frame_key not in frames:
            raise KeyError(f"Missing coordinate metadata for {frame_key}")
        normalized_labels = []
        for label in labels:
            normalized = dict(label)
            normalized["species"] = normalize_species(label.get("species", ""))
            normalized_labels.append(normalized)
        validate_labels(normalized_labels, frame_key)
        species = [label["species"] for label in normalized_labels]
        missing = [name for name in species if name not in global_classes]
        if missing:
            raise ValueError(f"Species missing from global classes: {sorted(set(missing))}")
        effective_records.append(
            {
                "frame": frames[frame_key],
                "labels": normalized_labels,
                "species": species,
                "preprocess": current.get("preprocess") or {},
                "note": current.get("note", ""),
                "source_kind": "draft" if drafts.get(frame_key) else "review",
            }
        )

    used_species = sorted(
        {species for record in effective_records for species in record["species"]},
        key=global_classes.index,
    )
    local_class_ids = {species: index for index, species in enumerate(used_species)}
    block_assignment = choose_block_splits(effective_records, args.seed, stream_ids)

    manifest_rows: list[dict[str, Any]] = []
    link_modes = Counter()
    split_frames = Counter()
    split_instances = Counter()
    split_blocks: dict[str, set[str]] = defaultdict(set)
    species_split_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for split in SPLIT_RATIOS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output / "records" / split).mkdir(parents=True, exist_ok=True)

    for record in sorted(
        effective_records, key=lambda value: (value["frame"].stream_id, value["frame"].seq_id)
    ):
        frame: FrameInfo = record["frame"]
        split = block_assignment[frame.route_block_id]
        stem = frame.frame_key
        image_target = output / "images" / split / f"{stem}.jpg"
        label_target = output / "labels" / split / f"{stem}.txt"
        record_target = output / "records" / split / f"{stem}.json"
        if not frame.source_image.is_file():
            raise FileNotFoundError(frame.source_image)
        link_modes[link_or_copy(frame.source_image, image_target)] += 1

        yolo_lines: list[str] = []
        exported_labels: list[dict[str, Any]] = []
        for label in record["labels"]:
            species = label["species"]
            local_id = local_class_ids[species]
            values = [str(local_id)]
            for x, y in label["points"]:
                values.extend((f"{float(x):.8f}", f"{float(y):.8f}"))
            yolo_lines.append(" ".join(values))
            exported_labels.append(
                {
                    **label,
                    "dataset_class_id": local_id,
                    "global_class_id": global_classes.index(species),
                }
            )
            species_split_counts[species][split] += 1
        label_target.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        preprocess = record["preprocess"]
        record_target.write_text(
            json.dumps(
                {
                    "frame_key": frame.frame_key,
                    "frame_id": frame.frame_id,
                    "stream_id": frame.stream_id,
                    "split": split,
                    "route_distance_m": frame.route_distance_m,
                    "route_block_id": frame.route_block_id,
                    "source_image": str(frame.source_image),
                    "linked_image": str(image_target),
                    "labels": exported_labels,
                    "preprocess": preprocess,
                    "note": record["note"],
                    "source_kind": record["source_kind"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        split_frames[split] += 1
        split_instances[split] += len(exported_labels)
        split_blocks[split].add(frame.route_block_id)
        manifest_rows.append(
            {
                "split": split,
                "frame_key": frame.frame_key,
                "stream_id": frame.stream_id,
                "frame_id": frame.frame_id,
                "source_kind": record["source_kind"],
                "source_image": str(frame.source_image),
                "image": str(image_target),
                "label": str(label_target),
                "tree_count": len(exported_labels),
                "route_distance_m": round(frame.route_distance_m, 6),
                "route_block_id": frame.route_block_id,
                "hong_kong_datetime": frame.hong_kong_datetime,
                "longitude": frame.longitude,
                "latitude": frame.latitude,
                "exposure_ev": preprocess.get("exposure_ev", 0.0),
                "contrast": preprocess.get("contrast", 1.0),
                "preprocess_reason": preprocess.get("reason", "normal"),
                "human_accepted": preprocess.get("human_accepted", False),
            }
        )

    with (output / "manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_map = [
        {
            "dataset_class_id": local_class_ids[species],
            "global_class_id": global_classes.index(species),
            "species": species,
        }
        for species in used_species
    ]
    (output / "class_map.json").write_text(
        json.dumps(class_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    data_yaml = [
        f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    data_yaml.extend(
        f"  {index}: {json.dumps(species, ensure_ascii=False)}"
        for index, species in enumerate(used_species)
    )
    (output / "data.yaml").write_text("\n".join(data_yaml) + "\n", encoding="utf-8")

    summary = {
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "input_precedence": "frame_drafts.json overrides frame_review_state.json",
        "unlabeled_frames_excluded": True,
        "images": len(effective_records),
        "instances": sum(len(record["labels"]) for record in effective_records),
        "used_classes": len(used_species),
        "global_classes": len(global_classes),
        "link_modes": dict(link_modes),
        "split_frames": dict(split_frames),
        "split_instances": dict(split_instances),
        "split_blocks": {
            split: sorted(blocks) for split, blocks in split_blocks.items()
        },
        "species_split_instances": {
            species: dict(counts) for species, counts in species_split_counts.items()
        },
        "class_map": class_map,
    }
    (output / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
