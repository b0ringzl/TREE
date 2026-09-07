"""Create the standard PTv2 manifest/classes view over C1 shared assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    source_manifest_path = root / "shared_manifest.json"
    source_classes_path = root / "classes_19.json"
    source_validation_path = root / "validation.json"
    for path in (
        source_manifest_path,
        source_classes_path,
        source_validation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    classes = json.loads(source_classes_path.read_text(encoding="utf-8"))
    validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError("C1 dataset validation did not pass")
    if len(classes) != 19:
        raise ValueError("C2a requires the 19-class definition")

    records = []
    split_histogram: Counter[str] = Counter()
    for source in source_manifest["records"]:
        record = dict(source)
        record["split"] = str(source["benchmark_split"])
        records.append(record)
        split_histogram[record["split"]] += 1
    records.sort(key=lambda record: str(record["sample_key"]))
    expected = {"train": 13116, "val": 570, "test": 3448}
    if dict(split_histogram) != expected:
        raise ValueError(
            f"Unexpected C2a split histogram: {dict(split_histogram)}"
        )

    manifest = {
        "format_version": 1,
        "stage": "C2a",
        "status": "complete",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "point_count_per_sample": 8192,
        "point_npz_key": "points_xyz",
        "image_size": [768, 512],
        "summary": {
            "sample_count": len(records),
            "class_count": len(classes),
            "split_histogram": expected,
            "physical_assets_copied": False,
        },
        "records": records,
    }
    manifest_path = root / "manifest.json"
    classes_path = root / "classes.json"
    atomic_json(manifest_path, manifest)
    atomic_json(classes_path, classes)
    result = {
        "stage": "C2a",
        "status": "passed",
        "dataset_root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "classes": str(classes_path),
        "classes_sha256": sha256_file(classes_path),
        "summary": manifest["summary"],
    }
    atomic_json(root / "ptv2_view_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
