#!/usr/bin/env python3
"""Merge fine-leaf and weeping fig labels into one 榕树 class safely."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELER_ROOT = PROJECT_ROOT / "derived" / "exhaustive_species_frame_labeler"
TARGET = "榕树"
MERGED_NAMES = (
    "Ficus microcarpa 榕樹(細葉榕)",
    "Ficus microcarpa 榕树(细叶榕)",
    "Ficus microcarpa 細葉榕",
    "Ficus microcarpa 细叶榕",
    "Ficus benjamina 垂葉榕",
    "Ficus benjamina 垂叶榕",
    "Ficus microcarpa",
    "Ficus benjamina",
    "榕樹(細葉榕)",
    "榕树(细叶榕)",
    "細葉榕",
    "细叶榕",
    "垂葉榕",
    "垂叶榕",
)


def is_merged_species(value: object) -> bool:
    name = str(value or "").strip()
    folded = name.casefold()
    return (
        folded.startswith("ficus microcarpa")
        or folded.startswith("ficus benjamina")
        or any(token in name for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕"))
    )


def normalize_species(value: object) -> str:
    name = str(value or "").strip()
    return TARGET if is_merged_species(name) or name == "榕樹" else name


def normalize_text(value: str) -> str:
    result = value
    for old in MERGED_NAMES:
        result = result.replace(old, TARGET)
    return result


def confidence_from_label(label: dict[str, Any]) -> float | None:
    current = label.get("confidence")
    if current is not None:
        try:
            value = float(current)
            if 0.0 <= value <= 1.0:
                return value
        except (TypeError, ValueError):
            pass

    species = normalize_species(label.get("species"))
    assistant = label.get("assistant_prelabel") or {}
    for candidate_key, confidence_key in (
        ("species", "domain_confidence"),
        ("domain_species", "domain_confidence"),
        ("web_species", "web_confidence"),
    ):
        if normalize_species(assistant.get(candidate_key)) != species:
            continue
        try:
            value = float(assistant.get(confidence_key))
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            return value

    note = str(label.get("note") or "")
    for marker in (r"域内YOLO候选[:：]", r"域内="):
        match = re.search(marker + r"\s*([^；|]+)", note)
        if not match:
            continue
        candidate_with_score = match.group(1).strip()
        scored = re.match(r"^(.+?)\s*\(([01](?:\.\d+)?)\)\s*$", candidate_with_score)
        if not scored or normalize_species(scored.group(1)) != species:
            continue
        value = float(scored.group(2))
        if 0.0 <= value <= 1.0:
            return value
    return None


def normalize_value(value: Any, counters: Counter[str]) -> Any:
    if isinstance(value, dict):
        result = {key: normalize_value(item, counters) for key, item in value.items()}
        if "species" in result:
            old_species = str(result["species"] or "")
            new_species = normalize_species(old_species)
            if new_species != old_species:
                counters["merged_species_fields"] += 1
            result["species"] = new_species
        return result
    if isinstance(value, list):
        return [normalize_value(item, counters) for item in value]
    if isinstance(value, str):
        normalized = normalize_text(value)
        if normalized != value:
            counters["normalized_text_fields"] += 1
        return normalized
    return value


def normalize_labels(value: Any, classes: list[str], counters: Counter[str]) -> Any:
    value = normalize_value(value, counters)

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            labels = item.get("labels")
            if isinstance(labels, list):
                for label in labels:
                    if not isinstance(label, dict) or "species" not in label:
                        continue
                    species = normalize_species(label["species"])
                    label["species"] = species
                    if species not in classes:
                        classes.append(species)
                    if "class_id" in label:
                        label["class_id"] = classes.index(species)
                        counters["class_ids_rewritten"] += 1
                    confidence = confidence_from_label(label)
                    if confidence is not None:
                        if label.get("confidence") != confidence:
                            counters["confidences_added"] += 1
                        label["confidence"] = confidence
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def remap_yolo_file(path: Path, old_to_new: dict[int, int]) -> str:
    output: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        old_id = int(parts[0])
        if old_id not in old_to_new:
            raise ValueError(f"Unknown class id {old_id} in {path}")
        parts[0] = str(old_to_new[old_id])
        output.append(" ".join(parts))
    return "\n".join(output) + ("\n" if output else "")


def backup_active_data(root: Path, timestamp: str) -> Path:
    backup = root / "runtime" / "backups" / f"banyan_merge_{timestamp}"
    if backup.exists():
        raise FileExistsError(backup)
    backup.mkdir(parents=True)
    for source, relative in (
        (root / "annotations" / "classes.json", Path("annotations/classes.json")),
        (root / "annotations" / "records", Path("annotations/records")),
        (root / "annotations" / "labels", Path("annotations/labels")),
        (root / "runtime" / "frame_drafts.json", Path("runtime/frame_drafts.json")),
        (root / "runtime" / "frame_review_state.json", Path("runtime/frame_review_state.json")),
    ):
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        elif source.is_file():
            shutil.copy2(source, target)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler-root", type=Path, default=DEFAULT_LABELER_ROOT)
    parser.add_argument("--apply", action="store_true", help="write changes after making a backup")
    cfg = parser.parse_args()
    root = cfg.labeler_root.resolve()
    classes_path = root / "annotations" / "classes.json"
    if not classes_path.is_file():
        raise FileNotFoundError(classes_path)

    old_classes = json.loads(classes_path.read_text(encoding="utf-8"))
    classes: list[str] = []
    for name in old_classes:
        normalized = normalize_species(name)
        if normalized and normalized not in classes:
            classes.append(normalized)
    old_to_new = {index: classes.index(normalize_species(name)) for index, name in enumerate(old_classes)}

    counters: Counter[str] = Counter()
    json_values: dict[Path, Any] = {}
    json_paths = [
        root / "runtime" / "frame_drafts.json",
        root / "runtime" / "frame_review_state.json",
        *sorted((root / "annotations" / "records").glob("**/*.json")),
    ]
    for path in json_paths:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        json_values[path] = normalize_labels(value, classes, counters)
        counters["json_files_scanned"] += 1

    yolo_values: dict[Path, str] = {}
    for path in sorted((root / "annotations" / "labels").glob("**/*.txt")):
        yolo_values[path] = remap_yolo_file(path, old_to_new)
        counters["yolo_files_remapped"] += 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "status": "applied" if cfg.apply else "dry_run",
        "timestamp_utc": timestamp,
        "labeler_root": str(root),
        "target_species": TARGET,
        "old_class_count": len(old_classes),
        "new_class_count": len(classes),
        "target_class_id": classes.index(TARGET),
        **counters,
    }

    if cfg.apply:
        backup = backup_active_data(root, timestamp)
        write_json(classes_path, classes)
        for path, value in json_values.items():
            write_json(path, value)
        for path, value in yolo_values.items():
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(value, encoding="utf-8")
            temporary.replace(path)
        report["backup"] = str(backup)
        report_path = PROJECT_ROOT / "reports" / f"{timestamp}_榕树类别合并报告.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, report)
        report["report"] = str(report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
