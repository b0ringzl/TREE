import argparse
import csv
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import laspy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PCTREES_ROOT = Path(__file__).resolve().parent / "pctrees_github"
TREE_ROOT = Path(__file__).resolve().parent.parent
PCT_ROOT = PCTREES_ROOT / "PCT_Pytorch"
POINTNET2_ROOT = PCT_ROOT / "pointnet2_ops_lib"
sys.path.insert(0, str(PCTREES_ROOT))
sys.path.insert(0, str(POINTNET2_ROOT))
sys.path.insert(0, str(PCT_ROOT))

from PCT_Pytorch.model import Pct
import point_cache
import utils


DEFAULT_TEST_DIR = TREE_ROOT / "lidar data" / "test"
DEFAULT_CACHE_DIR = TREE_ROOT / "lidar data" / "pctrees_test_cache_4096"
DEFAULT_MODEL_PATH = PCTREES_ROOT / "checkpoints" / "tree_lidar_pct_full_20260704_100ep_cache4096" / "models" / "model.t7"
DEFAULT_METRICS = PCTREES_ROOT / "checkpoints" / "tree_lidar_pct_full_20260704_100ep_cache4096" / "figures" / "metrics_summary.json"
DEFAULT_OUTPUT = PCTREES_ROOT / "checkpoints" / "tree_lidar_pct_full_20260704_100ep_cache4096" / "test_predictions_codabench.csv"


def parse_tree_id(path):
    match = re.fullmatch(r"(?:treeID_)?0*(\d+)\.la[sz]", Path(path).name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse tree ID from {path}")
    return int(match.group(1))


def collect_test_sources(test_dir):
    sources = []
    for filepath in sorted(Path(test_dir).glob("*.la[sz]")):
        sources.append((parse_tree_id(filepath), filepath))
    return sources


def load_class_names(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    class_names = payload.get("class_names")
    if not class_names:
        raise ValueError(f"No class_names found in {path}")
    return list(class_names)


def build_test_cache_one(task):
    tree_id = int(task["tree_id"])
    source_path = Path(task["source_path"])
    cache_dir = Path(task["cache_dir"])
    cache_path = cache_dir / f"{tree_id:05d}.pt"
    if cache_path.exists() and not task["overwrite"]:
        return {"treeID": tree_id, "cache": str(cache_path), "status": "reused"}

    try:
        las = laspy.read(source_path)
        points, _ = utils.las_to_pc(las)
        point_cache.write_point_cache(
            tree_id=tree_id,
            points=points,
            source_path=source_path,
            output_dir=cache_dir,
            cache_points=task["cache_points"],
            seed=task["seed"] + tree_id,
        )
        return {
            "treeID": tree_id,
            "cache": str(cache_path),
            "status": "built",
            "num_source_points": int(len(points)),
        }
    except Exception as exc:
        return {"treeID": tree_id, "source": str(source_path), "status": "failed", "error": repr(exc)}


def ensure_test_cache(sources, cache_dir, cache_points, seed, workers, overwrite):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "tree_id": tree_id,
            "source_path": str(source_path),
            "cache_dir": str(cache_dir),
            "cache_points": cache_points,
            "seed": seed,
            "overwrite": overwrite,
        }
        for tree_id, source_path in sources
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_test_cache_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="building test cache", unit="tree"):
            rows.append(future.result())

    index_path = cache_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["treeID", "cache", "status", "num_source_points", "source", "error"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["treeID"])):
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"Failed to cache {len(failed)} test files. First failure: {failed[0]}")
    return rows


class CachedTestTrees(Dataset):
    def __init__(self, sources, cache_dir, num_points):
        self.items = [(tree_id, Path(cache_dir) / f"{tree_id:05d}.pt") for tree_id, _ in sorted(sources)]
        self.num_points = int(num_points)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        tree_id, cache_path = self.items[index]
        payload = torch.load(cache_path, map_location="cpu")
        points = payload["points"] if isinstance(payload, dict) else payload
        return int(tree_id), points.clone().float()[: self.num_points]


def load_model(model_path, class_count, dropout, device):
    args = SimpleNamespace(dropout=dropout, output_channels=class_count)
    model = nn.DataParallel(Pct(args)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, dataset, class_names, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions = []
    with torch.no_grad():
        for tree_ids, points in tqdm(loader, desc="predicting test trees", unit="batch"):
            points = points.float().to(device).permute(0, 2, 1)
            logits = model(points)
            pred_ids = logits.argmax(dim=1).detach().cpu().tolist()
            for tree_id, pred_id in zip(tree_ids.tolist(), pred_ids):
                predictions.append({"treeID": int(tree_id), "predicted_species": class_names[int(pred_id)]})
    return predictions


def write_submission(predictions, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(sorted(predictions, key=lambda item: int(item["treeID"])))
    frame = frame[["treeID", "predicted_species"]]
    frame.to_csv(output_csv, index=False, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict hidden FOR-species20K test labels with trained PCTree model")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-points", type=int, default=4096)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sources = collect_test_sources(args.test_dir)
    if not sources:
        raise RuntimeError(f"No .las/.laz files found under {args.test_dir}")
    class_names = load_class_names(args.metrics_json)
    ensure_test_cache(
        sources=sources,
        cache_dir=args.cache_dir,
        cache_points=args.cache_points,
        seed=args.seed,
        workers=args.cache_workers,
        overwrite=args.overwrite_cache,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.model_path, len(class_names), args.dropout, device)
    dataset = CachedTestTrees(sources, args.cache_dir, args.num_points)
    predictions = predict(model, dataset, class_names, args.batch_size, args.num_workers, device)
    write_submission(predictions, args.output_csv)
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "test_dir": str(args.test_dir),
                "cache_dir": str(args.cache_dir),
                "test_files": len(sources),
                "predictions": len(predictions),
                "class_count": len(class_names),
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
