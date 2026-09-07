from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


UNKNOWN = "Unknown / 待定"
GLOBAL_ADDITIONS = (
    "Araucaria heterophylla 异叶南洋杉",
    "Alstonia scholaris 糖膠樹",
    "Lagerstroemia indica 紫薇",
    "Ravenala madagascariensis 旅人蕉",
)
NOTE_SPECIES = {
    "Alstonia scholaris": "Alstonia scholaris 糖膠樹",
    "紫薇": "Lagerstroemia indica 紫薇",
    "旅人蕉": "Ravenala madagascariensis 旅人蕉",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def orientation(a: list[float], b: list[float], c: list[float]) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 1 if value > 1e-12 else -1 if value < -1e-12 else 0


def crossing_pair(points: list[list[float]]) -> tuple[int, int] | None:
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
                return first, second
    return None


def uncross(points: list[list[float]]) -> tuple[list[list[float]], int]:
    result = [list(point) for point in points]
    repairs = 0
    while True:
        pair = crossing_pair(result)
        if pair is None:
            return result, repairs
        first, second = pair
        result[first + 1 : second + 1] = reversed(result[first + 1 : second + 1])
        repairs += 1
        if repairs > len(result) * len(result):
            raise RuntimeError("Polygon uncrossing did not converge")


def bbox(points: list[list[float]]) -> dict[str, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "x_center": (x_min + x_max) / 2,
        "y_center": (y_min + y_max) / 2,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def repair_state(state: dict[str, Any], classes: list[str]) -> tuple[Counter[str], int]:
    renamed: Counter[str] = Counter()
    polygon_repairs = 0
    for record in state.values():
        note = str(record.get("note", "")).strip()
        replacement = NOTE_SPECIES.get(note)
        for label in record.get("labels") or []:
            if replacement and str(label.get("species", "")).strip() == UNKNOWN:
                label["species"] = replacement
                renamed[replacement] += 1
            points, repairs = uncross(label.get("points") or [])
            if repairs:
                label["points"] = points
                polygon_repairs += 1
            species = str(label.get("species", "")).strip()
            if species not in classes:
                raise ValueError(f"Species missing from synchronized classes: {species}")
            label["class_id"] = classes.index(species)
            label["bbox"] = bbox(label["points"])
    return renamed, polygon_repairs


def sync_classes(paths: tuple[Path, ...]) -> list[str]:
    class_lists = [read_json(path) for path in paths]
    shortest = min(len(value) for value in class_lists)
    prefix = class_lists[0][:shortest]
    if any(value[:shortest] != prefix for value in class_lists[1:]):
        raise ValueError("Existing global class prefixes differ")
    classes = list(max(class_lists, key=len))
    for species in GLOBAL_ADDITIONS:
        if species not in classes:
            classes.append(species)
    for path in paths:
        atomic_json(path, classes)
    return classes


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    hewentian = project / "derived" / "hewentian_frame_labeler"
    jianshazui = project / "derived" / "jianshazui_frame_labeler"
    classes = sync_classes(
        (
            hewentian / "annotations" / "classes.json",
            jianshazui / "annotations" / "classes.json",
        )
    )

    draft_path = hewentian / "runtime" / "frame_drafts.json"
    review_path = hewentian / "runtime" / "frame_review_state.json"
    drafts = read_json(draft_path)
    reviews = read_json(review_path)
    draft_renamed, draft_repairs = repair_state(drafts, classes)
    review_renamed, review_repairs = repair_state(reviews, classes)
    atomic_json(draft_path, drafts)
    atomic_json(review_path, reviews)

    # Keep formal review artifacts consistent with the repaired runtime state.
    records_dir = hewentian / "annotations" / "records"
    labels_dir = hewentian / "annotations" / "labels"
    for frame_key, record in reviews.items():
        atomic_json(records_dir / f"{frame_key}.json", record)
        labels = record.get("labels") or []
        label_path = labels_dir / f"{frame_key}.txt"
        if labels:
            lines = []
            for label in labels:
                values = [str(label["class_id"])]
                for x, y in label["points"]:
                    values.extend((f"{float(x):.8f}", f"{float(y):.8f}"))
                lines.append(" ".join(values))
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif label_path.exists():
            label_path.unlink()

    renamed = draft_renamed + review_renamed
    print(
        json.dumps(
            {
                "global_classes": len(classes),
                "renamed_instances": dict(renamed),
                "renamed_total": sum(renamed.values()),
                "polygons_repaired": draft_repairs + review_repairs,
                "draft_records": len(drafts),
                "review_records": len(reviews),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
