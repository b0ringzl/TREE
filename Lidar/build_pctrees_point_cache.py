import argparse
import csv
from pathlib import Path
import re

import laspy
import pandas as pd
from tqdm import tqdm

PCTREES_ROOT = Path(__file__).resolve().parent / "pctrees_github"
import sys

sys.path.insert(0, str(PCTREES_ROOT))

import point_cache
import utils


DEFAULT_LABELS = Path(r"D:\TREE\lidar data\labels_pctrees_train.csv")
DEFAULT_DATA_DIR = Path(r"D:\TREE\lidar data\train")
DEFAULT_CACHE_DIR = Path(r"D:\TREE\lidar data\pctrees_cache_4096")


def collect_lidar_sources(data_dir):
    source_by_id = {}
    pattern = re.compile(r"(?:treeID_)?0*(\d+)\.la[sz]$", re.IGNORECASE)
    for filepath in sorted(Path(data_dir).glob("*.la[sz]")):
        match = pattern.match(filepath.name)
        if match:
            source_by_id[int(match.group(1))] = filepath
    return source_by_id


def build_cache(args):
    labels = pd.read_csv(args.labels)
    labels["tree_id"] = labels["tree_id"].astype(int)
    sources = collect_lidar_sources(args.data_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.cache_dir / "index.csv"

    rows = []
    built = skipped = reused = 0
    work = labels.itertuples(index=False)
    if args.max_files:
        work = list(work)[: args.max_files]

    for row in tqdm(work, desc="building point cache", unit="tree"):
        tree_id = int(row.tree_id)
        label = getattr(row, "label")
        source_path = sources.get(tree_id)
        if not source_path:
            skipped += 1
            rows.append({"tree_id": tree_id, "label": label, "status": "missing_source"})
            continue

        source_points = int(getattr(row, "num_points", 0) or 0)
        if source_points and source_points < args.min_points:
            skipped += 1
            rows.append({
                "tree_id": tree_id,
                "label": label,
                "source": source_path.name,
                "num_source_points": source_points,
                "status": "too_few_points",
            })
            continue

        cache_path = args.cache_dir / f"{tree_id:05d}.pt"
        if cache_path.exists() and not args.overwrite:
            reused += 1
            rows.append({
                "tree_id": tree_id,
                "label": label,
                "source": source_path.name,
                "cache": cache_path.name,
                "num_source_points": source_points,
                "cache_points": args.cache_points,
                "status": "reused",
            })
            continue

        las = laspy.read(source_path)
        points, _ = utils.las_to_pc(las)
        if len(points) < args.min_points:
            skipped += 1
            rows.append({
                "tree_id": tree_id,
                "label": label,
                "source": source_path.name,
                "num_source_points": len(points),
                "status": "too_few_points",
            })
            continue

        point_cache.write_point_cache(
            tree_id=tree_id,
            points=points,
            source_path=source_path,
            output_dir=args.cache_dir,
            cache_points=args.cache_points,
            seed=args.seed + tree_id,
        )
        built += 1
        rows.append({
            "tree_id": tree_id,
            "label": label,
            "source": source_path.name,
            "cache": cache_path.name,
            "num_source_points": len(points),
            "cache_points": args.cache_points,
            "status": "built",
        })

    with index_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "tree_id",
            "label",
            "source",
            "cache",
            "num_source_points",
            "cache_points",
            "status",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    print(f"cache_dir={args.cache_dir}")
    print(f"index={index_path}")
    print(f"built={built} reused={reused} skipped={skipped}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build fixed-size pctrees point tensor cache")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-points", type=int, default=4096)
    parser.add_argument("--min-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--max-files", type=int, help="Optional smoke-test limit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
