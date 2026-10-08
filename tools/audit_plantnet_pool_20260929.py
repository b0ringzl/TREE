"""Explain the cumulative eligibility funnel for the 2026-09-29 paired audit."""

import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from compare_plantnet_demo_20260929 import CLASSES, DATA_ROOTS, EXCLUDE, TEST, TRAIN, read, sha256


def main():
    rows = read(TEST)
    names = set(read(CLASSES)["original_names"])
    train_groups = {row["group_id"] for row in read(TRAIN)}
    counts = Counter()
    eligible_groups = set()
    metadata_cache = {}
    for row in rows:
        counts["fixed_test_rows"] += 1
        species = row["species"]
        if species not in names or species in EXCLUDE or len(species.split()) != 2:
            continue
        counts["independent_binomial_labels"] += 1
        source = Path(row["source_image"])
        if not any(dataset in source.parents for dataset in DATA_ROOTS) or not source.is_file():
            continue
        counts["existing_in_scoped_inat_collections"] += 1
        if row["group_id"] in train_groups:
            continue
        counts["no_train_observation_overlap"] += 1
        photo = re.search(r"_photo_(\d+)\.", source.name)
        metadata = source.parent.parent / "metadata.jsonl"
        if not photo or not metadata.is_file():
            continue
        if metadata not in metadata_cache:
            metadata_cache[metadata] = {str(item.get("photo_id")): item
                                        for line in metadata.read_text(encoding="utf-8").splitlines()
                                        if (item := json.loads(line))}
        item = metadata_cache[metadata].get(photo.group(1))
        if not item:
            continue
        counts["matched_photo_metadata"] += 1
        if item.get("photo_license_code", "").lower() != "cc0":
            continue
        counts["cc0"] += 1
        if item.get("inat_taxon_name") != species or item.get("inat_quality_grade") != "research":
            continue
        counts["exact_research_grade_label"] += 1
        if sha256(source) != row["sha256"]:
            raise RuntimeError(f"Changed source image: {source}")
        counts["sha256_verified"] += 1
        with Image.open(source) as image:
            if image.getexif().get_ifd(0x8825):
                continue
            width, height = image.size
        counts["no_exif_gps"] += 1
        if min(width, height) < 224:
            continue
        counts["strict_candidate_images"] += 1
        eligible_groups.add(row["group_id"])
    counts["strict_candidate_observations"] = len(eligible_groups)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
