"""Convert the leakage-controlled ten-class manifest into YOLO Detect format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.is_file():
        return boxes
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        try:
            values = [float(value) for value in parts]
        except ValueError:
            continue
        if len(values) == 5:
            _, cx, cy, width, height = values
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                continue
            x0, y0, x1, y1 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        elif len(values) >= 7 and (len(values) - 1) % 2 == 0:
            coordinates = values[1:]
            if any(value < 0 or value > 1 for value in coordinates):
                continue
            xs, ys = coordinates[0::2], coordinates[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if (output_dir / "dataset_summary.json").exists():
        print(f"Dataset already built: {output_dir}")
        return

    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest is empty")

    class_by_index = {
        int(row["class_index"]): row["species"] for row in rows
    }
    if sorted(class_by_index) != list(range(10)):
        raise ValueError(f"Expected class indices 0..9, got {sorted(class_by_index)}")
    class_names = [class_by_index[index] for index in range(10)]

    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Unexpected split: {split}")
        grouped[(split, row["image_path"])].append(row)
        if row["source"] == "hk_streetview":
            group_splits[row["group_id"]].add(split)
    leaked_groups = {key: value for key, value in group_splits.items() if len(value) > 1}
    if leaked_groups:
        raise ValueError(f"Tree groups cross splits: {leaked_groups}")

    content_splits: defaultdict[str, set[str]] = defaultdict(set)
    link_modes: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    class_split_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    index_rows: list[dict[str, str | int]] = []

    for (split, relative_image), image_rows in sorted(grouped.items()):
        source_image = PROJECT_ROOT / relative_image
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        content_hash = sha256_file(source_image)
        content_splits[content_hash].add(split)
        stem = hashlib.sha1(relative_image.encode("utf-8")).hexdigest()[:20]
        destination_image = output_dir / "images" / split / f"{stem}{source_image.suffix.lower()}"
        destination_label = output_dir / "labels" / split / f"{stem}.txt"
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        link_modes[link_or_copy(source_image, destination_image)] += 1

        parsed = source_boxes(PROJECT_ROOT / image_rows[0]["label_path"])
        yolo_lines: list[str] = []
        for row in sorted(image_rows, key=lambda item: int(item["annotation_index"])):
            annotation_index = int(row["annotation_index"])
            if annotation_index >= len(parsed):
                raise ValueError(f"Annotation index mismatch: {row}")
            x0, y0, x1, y1 = parsed[annotation_index]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            width, height = x1 - x0, y1 - y0
            class_index = int(row["class_index"])
            yolo_lines.append(
                f"{class_index} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
            )
            box_counts[split] += 1
            class_split_counts[row["species"]][split] += 1
        destination_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        image_counts[split] += 1
        first = image_rows[0]
        index_rows.append(
            {
                "split": split,
                "source": first["source"],
                "group_id": first["group_id"],
                "source_image": relative_image,
                "yolo_image": destination_image.relative_to(output_dir).as_posix(),
                "box_count": len(yolo_lines),
                "sha256": content_hash,
            }
        )

    content_leaks = {key: value for key, value in content_splits.items() if len(value) > 1}
    if content_leaks:
        raise ValueError(f"Exact image content crosses splits: {content_leaks}")
    for split in SPLITS:
        present = {
            species for species, counts in class_split_counts.items() if counts[split] > 0
        }
        if present != set(class_names):
            raise ValueError(f"{split} is missing classes: {sorted(set(class_names) - present)}")

    yaml_payload = {
        "path": output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(yaml_payload, stream, allow_unicode=True, sort_keys=False)

    with (output_dir / "image_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    summary = {
        "schema_version": 1,
        "status": "validated",
        "task": "YOLO11 detect, ten tree species",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "class_names": class_names,
        "image_counts": dict(image_counts),
        "box_counts": dict(box_counts),
        "class_split_box_counts": {
            species: dict(counts) for species, counts in class_split_counts.items()
        },
        "materialization": dict(link_modes),
        "checks": {
            "ten_classes_in_every_split": True,
            "hk_group_leakage": False,
            "exact_content_leakage": False,
            "holdout_is_hk_only": all(
                row["source"] == "hk_streetview"
                for row in rows
                if row["split"] in {"val", "test"}
            ),
        },
        "annotation_policy": "Original YOLO boxes retained; polygon annotations converted to their tight bounding box.",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
