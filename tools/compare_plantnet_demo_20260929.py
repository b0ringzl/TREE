"""Prepare a reproducible, rights-filtered paired Pl@ntNet/Demo audit.

This is read-only with respect to datasets, model weights, and deployments.
The held-out split has already been inspected, so the output is a regression
comparison, not a new independent accuracy estimate.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "outputs/SSTv1_Artocarpus_User_20260918_v1/test_rows.json"
TRAIN = ROOT / "outputs/SSTv1_Artocarpus_User_20260918_v1/train_rows.json"
CLASSES = ROOT / "outputs/SSTv1_Aquilaria_UserCrops_20260919_v2/classes.json"
DATA_ROOTS = [
    ROOT / "external_datasets/inat_max150",
    ROOT / "external_datasets/inaturalist_dated/2026-06-07",
    ROOT / "external_datasets/inat_parent_name_supplement",
]
DEFAULT_OUT = ROOT / "outputs/plantnet_demo_comparison_20260929_v1"
EXCLUDE = {
    "Ficus microcarpa", "Ficus benjamina",
    "Araucaria columnaris", "Araucaria heterophylla",
    "Bauhinia purpurea", "Bauhinia variegata", "Bauhinia x blakeana",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(count, seed, output, allow_repeated_observation=False):
    if output.exists():
        raise SystemExit(f"Output already exists; do not overwrite: {output}")
    test = read(TEST)
    train_groups = {r["group_id"] for r in read(TRAIN)}
    names = set(read(CLASSES)["original_names"])
    candidates = []
    for row in test:
        species = row["species"]
        if species not in names or species in EXCLUDE or len(species.split()) != 2:
            continue
        source = Path(row["source_image"])
        if not any(dataset in source.parents for dataset in DATA_ROOTS) or not source.is_file():
            continue
        if row["group_id"] in train_groups:
            raise RuntimeError(f"Train/test observation-group overlap: {row['group_id']}")
        photo = re.search(r"_photo_(\d+)\.", source.name)
        if not photo:
            continue
        metadata = source.parent.parent / "metadata.jsonl"
        if not metadata.is_file():
            continue
        match = None
        for line in metadata.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if str(item.get("photo_id")) == photo.group(1):
                match = item
                break
        if not match or match.get("photo_license_code", "").lower() != "cc0":
            continue
        if match.get("inat_taxon_name") != species or match.get("inat_quality_grade") != "research":
            continue
        if sha256(source) != row["sha256"]:
            raise RuntimeError(f"Changed source image: {source}")
        with Image.open(source) as image:
            if image.getexif().get_ifd(0x8825):
                continue  # Do not transmit embedded location metadata.
            width, height = image.size
        if min(width, height) < 224:
            continue
        candidates.append({
            "sample_id": row["sample_id"], "species": species,
            "group_id": row["group_id"], "image_path": str(source),
            "sha256": row["sha256"], "width": width, "height": height,
            "license": "CC0", "inat_observation": match["inat_uri"],
            "photo_id": photo.group(1),
        })
    # Simple random sample without replacement, with one image per source
    # observation. No model output is examined before this draw.
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = []
    seen_groups = set()
    for item in candidates:
        if item["group_id"] in seen_groups:
            continue
        chosen.append(item)
        seen_groups.add(item["group_id"])
        if len(chosen) == count:
            break
    unique_observations = len(chosen)
    if len(chosen) < count and allow_repeated_observation:
        # Keep the independently drawn one-per-observation prefix unchanged.
        # Extra photos are correlated and must be reported separately.
        seen_samples = {item["sample_id"] for item in chosen}
        for item in candidates:
            if item["sample_id"] in seen_samples:
                continue
            chosen.append(item)
            seen_samples.add(item["sample_id"])
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise RuntimeError(f"Only {len(chosen)} eligible samples, requested {count}")
    output.mkdir(parents=True)
    write(output / "manifest.json", {
        "scope": "Historically inspected classification test split; not an independent blind test",
        "sample_design": "Seeded random draw from CC0 iNaturalist research-grade images in the fixed test split; one photo per observation first, then explicitly flagged repeated-observation photos only if required; standard binomial species supported exactly by Demo",
        "seed": seed, "eligible_count": len(candidates), "count": count,
        "distinct_observations": len({item["group_id"] for item in chosen}),
        "one_photo_per_observation_prefix": unique_observations,
        "repeated_observation_photos_allowed": allow_repeated_observation,
        "excluded_combined_labels": sorted(EXCLUDE),
        "test_manifest": str(TEST), "test_manifest_sha256": sha256(TEST),
        "train_manifest": str(TRAIN), "train_manifest_sha256": sha256(TRAIN),
        "comparison_protocol": {
            "input": "Same original JPEG, one preselected plant target; no detection/SAM/crop",
            "demo": "Current official localhost:8043 API; street, image-only, photo geometry, context view",
            "plantnet": "User-specified World Flora browser UI; one image; organ auto unless required",
            "metric": "Exact scientific-name Top-1 agreement with iNaturalist source label; abstention/failed result is incorrect",
            "caveats": ["Source labels are not independent expert ground truth", "Test split has been inspected before", "Both systems may have seen public iNaturalist images during pretraining"],
        },
        "samples": chosen,
    })
    print(f"Prepared {count} / {len(candidates)} CC0 eligible test images: {output / 'manifest.json'}")
    for i, row in enumerate(chosen, 1):
        print(f"{i:02d} {row['species']}: {row['image_path']}")


def demo(output):
    manifest = read(output / "manifest.json")
    result_path = output / "demo_results_v2.json"
    if result_path.exists():
        raise SystemExit("Demo results already exist; do not overwrite")
    with requests.Session() as session:
        health = session.get("http://127.0.0.1:8043/api/health", timeout=30).json()
        if not health.get("ready") or health.get("classifier_alignment") != "aligned":
            raise RuntimeError(f"Demo is not ready/aligned: {health}")
        results = []
        for i, row in enumerate(manifest["samples"], 1):
            path = Path(row["image_path"])
            if sha256(path) != row["sha256"]:
                raise RuntimeError(f"Image changed before replay: {path}")
            with path.open("rb") as image:
                response = session.post(
                    "http://127.0.0.1:8043/api/predict",
                    files={"image": (path.name, image, "image/png" if path.suffix.lower() == ".png" else "image/jpeg")},
                    data={"scenario": "street", "source": "image", "projection": "photo", "view_policy": "context"},
                    timeout=180,
                )
            try:
                payload = response.json()
            except ValueError:
                payload = {"raw_response": response.text[:500]}
            results.append({"sample_id": row["sample_id"], "http_status": response.status_code, "response": payload})
            print(f"{i}/{len(manifest['samples'])}: {row['species']} HTTP {response.status_code}", flush=True)
    write(result_path, {"health": health, "results": results})
    print(result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "demo"])
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-repeated-observation", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "prepare":
        prepare(args.count, args.seed, output, args.allow_repeated_observation)
    else:
        demo(output)
