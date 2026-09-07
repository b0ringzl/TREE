from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CANONICAL_SPECIES = "Araucaria heterophylla 异叶南洋杉"
SPECIES_ALIASES = {
    "Araucaria heterophylla异叶南阳杉": CANONICAL_SPECIES,
    "Araucaria heterophylla 异叶南阳杉": CANONICAL_SPECIES,
    "Araucaria heterophylla异叶南洋杉": CANONICAL_SPECIES,
}


def orientation(a: list[float], b: list[float], c: list[float]) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if value > 1e-12:
        return 1
    if value < -1e-12:
        return -1
    return 0


def proper_intersection(
    a: list[float], b: list[float], c: list[float], d: list[float]
) -> bool:
    return (
        orientation(a, b, c) * orientation(a, b, d) < 0
        and orientation(c, d, a) * orientation(c, d, b) < 0
    )


def first_crossing(points: list[list[float]]) -> tuple[int, int] | None:
    count = len(points)
    for first in range(count):
        for second in range(first + 1, count):
            if second == (first + 1) % count or (second + 1) % count == first:
                continue
            if proper_intersection(
                points[first],
                points[(first + 1) % count],
                points[second],
                points[(second + 1) % count],
            ):
                return first, second
    return None


def untangle(points: list[list[float]]) -> tuple[list[list[float]], int]:
    repaired = [list(point) for point in points]
    operations = 0
    while operations <= len(repaired) * len(repaired):
        crossing = first_crossing(repaired)
        if crossing is None:
            return repaired, operations
        first, second = crossing
        repaired[first + 1 : second + 1] = reversed(repaired[first + 1 : second + 1])
        operations += 1
    raise RuntimeError("Polygon repair did not converge")


def bbox(points: list[list[float]]) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    minimum_x, maximum_x = min(xs), max(xs)
    minimum_y, maximum_y = min(ys), max(ys)
    return {
        "x_center": (minimum_x + maximum_x) / 2,
        "y_center": (minimum_y + maximum_y) / 2,
        "width": maximum_x - minimum_x,
        "height": maximum_y - minimum_y,
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def repair_store(
    records: dict[str, Any], classes: list[str]
) -> tuple[int, int, list[dict[str, Any]]]:
    renamed = 0
    repaired_count = 0
    repairs: list[dict[str, Any]] = []
    for frame_key, record in records.items():
        for label_number, label in enumerate(record.get("labels") or [], start=1):
            species = str(label.get("species", "")).strip()
            canonical = SPECIES_ALIASES.get(species, species)
            if canonical != species:
                label["species"] = canonical
                renamed += 1
            points = label.get("points") or []
            repaired, operation_count = untangle(points) if len(points) >= 3 else (points, 0)
            if operation_count:
                label["points"] = repaired
                if "bbox" in label:
                    label["bbox"] = bbox(repaired)
                repaired_count += 1
                repairs.append(
                    {
                        "frame_key": frame_key,
                        "label_number": label_number,
                        "label_id": label.get("label_id"),
                        "species": label.get("species"),
                        "two_opt_operations": operation_count,
                    }
                )
            if "class_id" in label:
                label["class_id"] = classes.index(label["species"])
    return renamed, repaired_count, repairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair Jianshazui annotation species and self-intersecting polygons."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    base = args.project_root / "derived" / "jianshazui_frame_labeler"
    drafts_path = base / "runtime" / "frame_drafts.json"
    review_path = base / "runtime" / "frame_review_state.json"
    classes_path = base / "annotations" / "classes.json"

    drafts = json.loads(drafts_path.read_text(encoding="utf-8"))
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if CANONICAL_SPECIES not in classes:
        classes.append(CANONICAL_SPECIES)

    draft_renamed, draft_repaired, draft_repairs = repair_store(drafts, classes)
    review_renamed, review_repaired, review_repairs = repair_store(reviews, classes)

    atomic_json(drafts_path, drafts)
    atomic_json(review_path, reviews)
    atomic_json(classes_path, classes)
    print(
        json.dumps(
            {
                "canonical_species": CANONICAL_SPECIES,
                "class_id": classes.index(CANONICAL_SPECIES),
                "renamed_labels": draft_renamed + review_renamed,
                "repaired_polygons": draft_repaired + review_repaired,
                "repairs": draft_repairs + review_repairs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
