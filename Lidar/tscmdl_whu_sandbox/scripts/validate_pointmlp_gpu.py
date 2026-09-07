"""Validate the compiled PointNet2 extension and PointMLP GPU forward passes."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
POINTMLP_ROOT = SANDBOX_ROOT / "vendor" / "pointMLP-pytorch"
POINTNET2_ROOT = POINTMLP_ROOT / "pointnet2_ops_lib"
MODEL_ROOT = POINTMLP_ROOT / "classification_ModelNet40"

sys.path.insert(0, str(POINTNET2_ROOT))
sys.path.insert(0, str(MODEL_ROOT))

from models.pointmlp import Model, pointMLP  # noqa: E402
from pointnet2_ops import pointnet2_utils  # noqa: E402


def timed_forward(model: torch.nn.Module, points: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(points)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    return output, elapsed, peak_mib


def make_pointmlp(points: int, num_classes: int = 40) -> Model:
    return Model(
        points=points,
        class_num=num_classes,
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    results: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }

    xyz = torch.rand(2, 256, 3, device=device, dtype=torch.float32).contiguous()
    fps_indices = pointnet2_utils.furthest_point_sample(xyz, 64)
    if fps_indices.shape != (2, 64):
        raise AssertionError(f"Unexpected FPS shape: {tuple(fps_indices.shape)}")
    if fps_indices.min().item() < 0 or fps_indices.max().item() >= xyz.shape[1]:
        raise AssertionError("FPS returned an out-of-range index")
    results["fps"] = {
        "shape": list(fps_indices.shape),
        "dtype": str(fps_indices.dtype),
        "min": int(fps_indices.min().item()),
        "max": int(fps_indices.max().item()),
    }

    small_model = pointMLP(num_classes=40).to(device).eval()
    small_input = torch.rand(1, 3, 1024, device=device)
    small_output, small_seconds, small_peak_mib = timed_forward(small_model, small_input)
    if small_output.shape != (1, 40) or not torch.isfinite(small_output).all():
        raise AssertionError("PointMLP 1024-point output is invalid")
    results["pointmlp_1024"] = {
        "output_shape": list(small_output.shape),
        "elapsed_seconds": round(small_seconds, 4),
        "peak_allocated_mib": round(small_peak_mib, 1),
    }
    del small_model, small_input, small_output
    torch.cuda.empty_cache()

    large_model = make_pointmlp(points=8192).to(device).eval()
    large_input = torch.rand(1, 3, 8192, device=device)
    large_output, large_seconds, large_peak_mib = timed_forward(large_model, large_input)
    if large_output.shape != (1, 40) or not torch.isfinite(large_output).all():
        raise AssertionError("PointMLP 8192-point output is invalid")
    results["pointmlp_8192"] = {
        "output_shape": list(large_output.shape),
        "elapsed_seconds": round(large_seconds, 4),
        "peak_allocated_mib": round(large_peak_mib, 1),
    }

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
