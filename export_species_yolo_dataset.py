"""Export per-species LabelPaw annotations to one YOLO dataset.

Supported source layouts:

    species_root/
      Species A/
        image_001.jpg
        image_001.txt
      Species B/
        image_001.jpg
        image_001.txt

or:

    species_root/
      Species A/
        images/
          image_001.jpg
          image_001.txt

By default, the exported class list is derived from D:\TREE\选取树种, and
source folder names are normalized with D:\TREE\tools\tree_species_label_merge_map.csv
when the merged label exists in that class root. This keeps LabelPaw exports in
sync with project-level species merges such as Bridelia tomentosa(Bridelia insulana).
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
DEFAULT_SPECIES_ROOT = Path(r"D:\TREE\external_datasets\inat_max150")
DEFAULT_OUTPUT = Path(r"D:\TREE\models\web_tree_species_seg_dataset_v1")
DEFAULT_CLASS_ROOT = Path(r"D:\TREE\选取树种")
DEFAULT_MERGE_MAP = Path(r"D:\TREE\tools\tree_species_label_merge_map.csv")


@dataclass(frozen=True)
class ExportItem:
    source_species: str
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


def iter_species_dirs(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name not in {"duplicate_photo_review"}
        ],
        key=lambda path: path.name.lower(),
    )


def load_class_names(class_root: Path | None) -> list[str]:
    if class_root is None or not class_root.exists():
        return []
    return [path.name for path in iter_species_dirs(class_root)]


def load_species_merge_map(merge_map_path: Path | None, allowed_labels: set[str] | None) -> dict[str, str]:
    if merge_map_path is None or not merge_map_path.exists():
        return {}

    merge_map: dict[str, str] = {}
    with merge_map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = (row.get("original_species") or "").strip()
            label = (row.get("label_species") or "").strip()
            if not source or not label:
                continue
            if allowed_labels is not None and label not in allowed_labels:
                continue
            merge_map[source] = label
    return merge_map


def normalize_species_name(source_name: str, merge_map: dict[str, str]) -> str:
    return merge_map.get(source_name, source_name)


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


def iter_species_image_paths(species_dir: Path) -> tuple[Path, list[Path], str]:
    """Return the image folder and image files for one species directory."""
    nested_images_dir = species_dir / "images"
    image_dir = nested_images_dir if nested_images_dir.is_dir() else species_dir
    image_paths = sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS],
        key=lambda path: path.name.lower(),
    )
    layout = "nested_images" if image_dir == nested_images_dir else "flat"
    return image_dir, image_paths, layout


def collect_items(
    species_root: Path,
    val_fraction: float,
    test_fraction: float,
    class_root: Path | None = DEFAULT_CLASS_ROOT,
    merge_map_path: Path | None = DEFAULT_MERGE_MAP,
) -> tuple[list[ExportItem], list[str], list[str]]:
    species_dirs = iter_species_dirs(species_root)
    classes = load_class_names(class_root)
    allowed_labels = set(classes) if classes else None
    merge_map = load_species_merge_map(merge_map_path, allowed_labels)

    if not classes:
        classes = sorted({normalize_species_name(path.name, merge_map) for path in species_dirs}, key=str.lower)
    class_to_idx = {name: index for index, name in enumerate(classes)}

    items: list[ExportItem] = []
    warnings: list[str] = []
    labeled_by_species: dict[str, list[tuple[str, Path, Path]]] = {name: [] for name in classes}

    for species_dir in species_dirs:
        label_species = normalize_species_name(species_dir.name, merge_map)
        if label_species not in class_to_idx:
            warnings.append(
                f"{species_dir.name}: skipped because normalized label '{label_species}' is not in class list"
            )
            continue
        image_dir, image_paths, layout = iter_species_image_paths(species_dir)
        if not image_paths:
            warnings.append(f"{species_dir.name}: no images found in {image_dir} ({layout})")
            continue
        labeled_images = [path for path in image_paths if path.with_suffix(".txt").exists()]
        missing_count = len(image_paths) - len(labeled_images)
        if missing_count:
            warnings.append(
                f"{species_dir.name}: skipped {missing_count} images without matching .txt labels in {image_dir}"
            )

        for image_path in labeled_images:
            labeled_by_species.setdefault(label_species, []).append(
                (species_dir.name, image_path, image_path.with_suffix(".txt"))
            )

    for label_species in classes:
        labeled_records = labeled_by_species.get(label_species, [])
        split_map = split_species_items([record[1] for record in labeled_records], val_fraction, test_fraction)
        class_id = class_to_idx[label_species]
        for source_species, image_path, label_path in labeled_records:
            split = split_map.get(image_path)
            if split is None:
                continue
            items.append(
                ExportItem(
                    source_species=source_species,
                    species=label_species,
                    class_id=class_id,
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
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
    class_root: Path | None = DEFAULT_CLASS_ROOT,
    merge_map_path: Path | None = DEFAULT_MERGE_MAP,
) -> dict:
    if not species_root.exists():
        raise FileNotFoundError(f"Species root not found: {species_root}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)

    items, classes, warnings = collect_items(
        species_root=species_root,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        class_root=class_root,
        merge_map_path=merge_map_path,
    )
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
                "source_species": item.source_species,
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
            "source_species",
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
        "class_root": str(class_root) if class_root else None,
        "merge_map": str(merge_map_path) if merge_map_path else None,
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
    parser.add_argument("--species-root", type=Path, default=DEFAULT_SPECIES_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--class-root",
        type=Path,
        default=DEFAULT_CLASS_ROOT,
        help="Directory whose child folders define the exported class list and class id order.",
    )
    parser.add_argument(
        "--merge-map",
        type=Path,
        default=DEFAULT_MERGE_MAP,
        help="CSV with original_species,label_species columns used to merge source folders into class labels.",
    )
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
        class_root=args.class_root,
        merge_map_path=args.merge_map,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
