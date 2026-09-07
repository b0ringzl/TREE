"""Run a PointMLP GPU forward pass on prepared A3 sample archives."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
POINTMLP_ROOT = SANDBOX_ROOT / "vendor" / "pointMLP-pytorch"
sys.path.insert(0, str(POINTMLP_ROOT / "pointnet2_ops_lib"))
sys.path.insert(0, str(POINTMLP_ROOT / "classification_ModelNet40"))

from models.pointmlp import Model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    return parser.parse_args()


def make_model() -> Model:
    return Model(
        points=8192,
        class_num=19,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="anchor",
        dim_expansion=[2, 2, 2, 2],
        pre_blocks=[2, 2, 2, 2],
        pos_blocks=[2, 2, 2, 2],
        k_neighbors=[24, 24, 24, 24],
        reducers=[2, 2, 2, 2],
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    sample_paths = sorted(args.sample_dir.glob("*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No NPZ samples found in {args.sample_dir}")

    points = []
    labels = []
    tree_ids = []
    for path in sample_paths:
        with np.load(path, allow_pickle=False) as archive:
            normalized = archive["points_xyz_normalized"]
            label = int(archive["benchmark_label_id"])
            if normalized.shape != (8192, 3) or normalized.dtype != np.float32:
                raise ValueError(f"Unexpected point tensor in {path}: {normalized.shape} {normalized.dtype}")
            if label < 0 or label >= 19:
                raise ValueError(f"Benchmark label is outside [0, 19): {label}")
            points.append(normalized)
            labels.append(label)
            tree_ids.append(int(archive["tree_id"]))

    batch = torch.from_numpy(np.stack(points)).permute(0, 2, 1).contiguous().cuda()
    model = make_model().cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if logits.shape != (len(sample_paths), 19) or not torch.isfinite(logits).all():
        raise AssertionError(f"Invalid PointMLP output: {tuple(logits.shape)}")

    result = {
        "sample_count": len(sample_paths),
        "tree_ids": tree_ids,
        "benchmark_label_ids": labels,
        "input_shape": list(batch.shape),
        "output_shape": list(logits.shape),
        "finite_output": True,
        "elapsed_seconds": round(elapsed, 4),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
