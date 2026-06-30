from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import laspy
import pandas as pd


def count_points(tree_id: int, laz_files: dict[int, Path]) -> tuple[int, int | None]:
    laz_path = laz_files.get(tree_id)
    if laz_path is None:
        return tree_id, None
    with laspy.open(laz_path) as reader:
        return tree_id, int(reader.header.point_count)


def build_labels(metadata_path: Path, data_dir: Path, output_path: Path, workers: int = 12) -> None:
    metadata = pd.read_csv(metadata_path)
    laz_files = {int(path.stem): path for path in data_dir.glob("*.laz") if path.stem.isdigit()}
    tree_ids = metadata["treeID"].astype(int).tolist()

    counts: dict[int, int | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(count_points, tree_id, laz_files) for tree_id in tree_ids]
        for future in as_completed(futures):
            tree_id, point_count = future.result()
            counts[tree_id] = point_count

    labels = metadata.rename(columns={"treeID": "tree_id", "species": "label"})[["tree_id", "label"]].copy()
    labels["tree_id"] = labels["tree_id"].astype(int)
    labels["num_points"] = labels["tree_id"].map(counts)
    labels = labels.dropna(subset=["num_points"])
    labels["num_points"] = labels["num_points"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False)
    print(f"wrote {output_path}")
    print(f"rows={len(labels)} classes={labels['label'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pctrees-compatible labels from LiDAR metadata.")
    parser.add_argument("--metadata", type=Path, default=Path(r"D:\TREE\lidar data\tree_metadata_dev.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\TREE\lidar data\train"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\TREE\lidar data\labels_pctrees_train.csv"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    build_labels(args.metadata, args.data_dir, args.output, args.workers)


if __name__ == "__main__":
    main()
