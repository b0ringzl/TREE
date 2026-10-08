import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
import sys

import laspy
import pandas as pd
from tqdm import tqdm


PCTREES_ROOT = Path(__file__).resolve().parent / "pctrees_github"
sys.path.insert(0, str(PCTREES_ROOT))

import point_cache
import utils


TREE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = TREE_ROOT / "lidar data" / "labels_pctrees_train.csv"
DEFAULT_DATA_DIR = TREE_ROOT / "lidar data" / "train"
DEFAULT_CACHE_DIR = TREE_ROOT / "lidar data" / "pctrees_cache_4096"


def collect_lidar_sources(data_dir):
    source_by_id = {}
    pattern = re.compile(r"(?:treeID_)?0*(\d+)\.la[sz]$", re.IGNORECASE)
    for filepath in sorted(Path(data_dir).glob("*.la[sz]")):
        match = pattern.match(filepath.name)
        if match:
            source_by_id[int(match.group(1))] = filepath
    return source_by_id


def _int_or_zero(value):
    if pd.isna(value):
        return 0
    return int(value)


def collect_cache_tasks(labels, sources, cache_dir, cache_points, min_points, seed, overwrite):
    tasks = []
    rows = []
    counts = {"reused": 0, "skipped": 0}
    cache_dir = Path(cache_dir)

    for row in labels.itertuples(index=False):
        tree_id = int(row.tree_id)
        label = getattr(row, "label")
        source_path = sources.get(tree_id)
        if not source_path:
            counts["skipped"] += 1
            rows.append({"tree_id": tree_id, "label": label, "status": "missing_source"})
            continue

        source_points = _int_or_zero(getattr(row, "num_points", 0))
        if source_points and source_points < min_points:
            counts["skipped"] += 1
            rows.append(
                {
                    "tree_id": tree_id,
                    "label": label,
                    "source": source_path.name,
                    "num_source_points": source_points,
                    "status": "too_few_points",
                }
            )
            continue

        cache_path = cache_dir / f"{tree_id:05d}.pt"
        if cache_path.exists() and not overwrite:
            counts["reused"] += 1
            rows.append(
                {
                    "tree_id": tree_id,
                    "label": label,
                    "source": source_path.name,
                    "cache": cache_path.name,
                    "num_source_points": source_points,
                    "cache_points": cache_points,
                    "status": "reused",
                }
            )
            continue

        tasks.append(
            {
                "tree_id": tree_id,
                "label": label,
                "source_path": str(source_path),
                "source_name": source_path.name,
                "source_points": source_points,
                "cache_dir": str(cache_dir),
                "cache_points": cache_points,
                "min_points": min_points,
                "seed": seed + tree_id,
                "overwrite": overwrite,
            }
        )

    return tasks, rows, counts


def build_one_cache(task):
    tree_id = int(task["tree_id"])
    cache_dir = Path(task["cache_dir"])
    source_path = Path(task["source_path"])
    cache_path = cache_dir / f"{tree_id:05d}.pt"

    if cache_path.exists() and not task["overwrite"]:
        return {
            "tree_id": tree_id,
            "label": task["label"],
            "source": task["source_name"],
            "cache": cache_path.name,
            "num_source_points": task["source_points"],
            "cache_points": task["cache_points"],
            "status": "reused",
        }

    try:
        las = laspy.read(source_path)
        points, _ = utils.las_to_pc(las)
        if len(points) < task["min_points"]:
            return {
                "tree_id": tree_id,
                "label": task["label"],
                "source": task["source_name"],
                "num_source_points": len(points),
                "status": "too_few_points",
            }

        point_cache.write_point_cache(
            tree_id=tree_id,
            points=points,
            source_path=source_path,
            output_dir=cache_dir,
            cache_points=task["cache_points"],
            seed=task["seed"],
        )
        return {
            "tree_id": tree_id,
            "label": task["label"],
            "source": task["source_name"],
            "cache": cache_path.name,
            "num_source_points": len(points),
            "cache_points": task["cache_points"],
            "status": "built",
        }
    except Exception as exc:
        return {
            "tree_id": tree_id,
            "label": task["label"],
            "source": task["source_name"],
            "status": "failed",
            "error": repr(exc),
        }


def write_index(index_path, rows):
    fieldnames = [
        "tree_id",
        "label",
        "source",
        "cache",
        "num_source_points",
        "cache_points",
        "status",
        "error",
    ]
    rows = sorted(rows, key=lambda item: int(item.get("tree_id", 0)))
    with Path(index_path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_cache(args):
    labels = pd.read_csv(args.labels)
    labels["tree_id"] = labels["tree_id"].astype(int)
    if args.max_files:
        labels = labels.iloc[: args.max_files].copy()

    sources = collect_lidar_sources(args.data_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.cache_dir / "index.csv"

    tasks, rows, initial_counts = collect_cache_tasks(
        labels=labels,
        sources=sources,
        cache_dir=args.cache_dir,
        cache_points=args.cache_points,
        min_points=args.min_points,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    built = failed = 0
    reused = initial_counts["reused"]
    skipped = initial_counts["skipped"]

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_one_cache, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="building point cache", unit="tree"):
                row = future.result()
                status = row.get("status")
                if status == "built":
                    built += 1
                elif status == "reused":
                    reused += 1
                elif status == "failed":
                    failed += 1
                else:
                    skipped += 1
                rows.append(row)

    write_index(index_path, rows)
    print(f"cache_dir={args.cache_dir}")
    print(f"index={index_path}")
    print(f"built={built} reused={reused} skipped={skipped} failed={failed} workers={args.workers}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build fixed-size pctrees point tensor cache in parallel")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-points", type=int, default=4096)
    parser.add_argument("--min-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, help="Optional smoke-test limit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
