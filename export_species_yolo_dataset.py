"""Export per-species LabelPaw annotations to one YOLO dataset.

The source layout is expected to be:

    species_root/
      Species A/
        image_001.jpg
        image_001.txt
      Species B/
        image_001.jpg
        image_001.txt

Each source folder is treated as one species class. Source label class ids are
rewritten to the global class id derived from the species folder name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ExportItem:
    species: str
    class_id: int
    image_path: Path
    label_path: Path
    split: str


def safe_component(value: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    text = re.sub(r"_+", "_", text)
    return (text or "item")[:max_len]


def short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def yolo_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def stable_sort_key(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def split_species_items(
    image_paths: list[Path],
    val_fraction: float,
    test_fraction: float,
) -> dict[Path, str]:
    """Deterministically split one species while keeping tiny classes usable."""
    ordered = sorted(
        image_paths,
        key=lambda path: short_hash(stable_sort_key(path), 16),
    )
    total = len(ordered)
    if total == 0:
        return {}

    test_count = int(round(total * test_fraction))
    val_count = int(round(total * val_fraction))

    if total >= 3 and val_fraction > 0:
        val_count = max(1, val_count)
    if total >= 10 and test_fraction > 0:
        test_count = max(1, test_count)

    if val_count + test_count >= total:
        overflow = val_count + test_count - (total - 1)
        if test_count >= overflow:
            test_count -= overflow
        else:
            overflow -= test_count
            test_count = 0
            val_count = max(0, val_count - overflow)

    result: dict[Path, str] = {}
    for index, path in enumerate(ordered):
        if index < test_count:
            split = "test"
        elif index < test_count + val_count:
            split = "val"
        else:
            split = "train"
        result[path] = split
    return result


def parse_and_remap_label(label_path: Path, class_id: int) -> tuple[list[str], list[str]]:
    output_lines: list[str] = []
    warnings: list[str] = []

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 7:
            warnings.append(f"{label_path}:{line_number}: skipped non-segmentation row with {len(parts)} fields")
            continue
        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            warnings.append(f"{label_path}:{line_number}: skipped row with non-numeric coordinates")
            continue
        if len(coords) % 2 != 0:
            warnings.append(f"{label_path}:{line_number}: skipped row with odd number of coordinates")
            continue
        coords = [max(0.0, min(1.0, value)) for value in coords]
        output_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in coords))

    return output_lines, warnings


def collect_items(
    species_root: Path,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[ExportItem], list[str], list[str]]:
    species_dirs = sorted([path for path in species_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    classes = [path.name for path in species_dirs]
    class_to_idx = {name: index for index, name in enumerate(classes)}

    items: list[ExportItem] = []
    warnings: list[str] = []

    for species_dir in species_dirs:
        image_paths = sorted(
            [path for path in species_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS],
            key=lambda path: path.name.lower(),
        )
        labeled_images = [path for path in image_paths if path.with_suffix(".txt").exists()]
        missing_count = len(image_paths) - len(labeled_images)
        if missing_count:
            warnings.append(f"{species_dir.name}: skipped {missing_count} images without matching .txt labels")

        split_map = split_species_items(labeled_images, val_fraction, test_fraction)
        for image_path in labeled_images:
            items.append(
                ExportItem(
                    species=species_dir.name,
                    class_id=class_to_idx[species_dir.name],
                    image_path=image_path,
                    label_path=image_path.with_suffix(".txt"),
                    split=split_map[image_path],
                )
            )

    return items, classes, warnings


def ensure_output_dirs(output: Path) -> None:
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output: Path, classes: list[str]) -> None:
    lines = [
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    for index, name in enumerate(classes):
        lines.append(f"  {index}: {yolo_quote(name)}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_dataset(
    species_root: Path,
    output: Path,
    val_fraction: float,
    test_fraction: float,
    overwrite: bool,
) -> dict:
    if not species_root.exists():
        raise FileNotFoundError(f"Species root not found: {species_root}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)

    items, classes, warnings = collect_items(species_root, val_fraction, test_fraction)
    ensure_output_dirs(output)

    manifest_rows: list[dict[str, str | int]] = []
    used_names: set[str] = set()
    exported = 0

    for index, item in enumerate(sorted(items, key=lambda value: (value.species.lower(), value.image_path.name.lower()))):
        label_lines, label_warnings = parse_and_remap_label(item.label_path, item.class_id)
        warnings.extend(label_warnings)
        if not label_lines:
            warnings.append(f"{item.label_path}: skipped because no valid polygon rows remained")
            continue

        species_part = safe_component(item.species, 48)
        stem_part = safe_component(item.image_path.stem, 48)
        hash_part = short_hash(stable_sort_key(item.image_path), 10)
        base_name = f"{exported:06d}__{species_part}__{stem_part}__{hash_part}"
        while base_name in used_names:
            base_name = f"{exported:06d}__{species_part}__{stem_part}__{hash_part}_{len(used_names)}"
        used_names.add(base_name)

        image_name = f"{base_name}{item.image_path.suffix.lower()}"
        label_name = f"{base_name}.txt"
        image_out = output / "images" / item.split / image_name
        label_out = output / "labels" / item.split / label_name

        shutil.copy2(item.image_path, image_out)
        label_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        manifest_rows.append(
            {
                "split": item.split,
                "class_id": item.class_id,
                "species": item.species,
                "source_image": str(item.image_path),
                "source_label": str(item.label_path),
                "export_image": str(image_out),
                "export_label": str(label_out),
                "export_stem": base_name,
                "polygon_rows": len(label_lines),
            }
        )
        exported += 1

    (output / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (output / "class_to_idx.json").write_text(
        json.dumps({name: index for index, name in enumerate(classes)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dataset_yaml(output, classes)

    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "class_id",
            "species",
            "source_image",
            "source_label",
            "export_image",
            "export_label",
            "export_stem",
            "polygon_rows",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    split_counts = {
        split: sum(1 for row in manifest_rows if row["split"] == split)
        for split in ("train", "val", "test")
    }
    report = {
        "species_root": str(species_root),
        "output": str(output),
        "classes": len(classes),
        "source_labeled_images": len(items),
        "exported_images": exported,
        "split_counts": split_counts,
        "warnings": warnings,
    }
    (output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-species LabelPaw labels to one YOLO segmentation dataset.")
    parser.add_argument("--species-root", type=Path, default=Path(r"D:\TREE\选取树种"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\TREE\models\web_tree_species_seg_dataset_v1"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = export_dataset(
        species_root=args.species_root,
        output=args.output,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
