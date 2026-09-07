"""Use the vendor-colorized mapped cloud to diagnose panorama handedness."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from diagnose_lidar_camera_axes import COLOR_DTYPE, read_binary_pcd, world_from_flu
from render_temporal_lidar_panorama import load_frames


PROJECT_ROOT = Path(__file__).resolve().parent
VMMS_ROOT = PROJECT_ROOT.parent.parent / "vmms" / "2023-10-27_hewentian"
DEFAULT_CLOUD = VMMS_ROOT / "pointcloud" / "ColorCloudPoint" / "4319_img4335_LiD102044_ColorizedCloud.pcd"
OUTPUT_ROOT = PROJECT_ROOT / "derived" / "temporal_lidar_projection"


def packed_rgb(records: np.ndarray) -> np.ndarray:
    packed = records["rgb"].astype(np.uint32)
    return np.column_stack(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)).astype(np.float32)


def nearest_per_pixel(pixel: np.ndarray, depth: np.ndarray) -> np.ndarray:
    order = np.lexsort((depth, pixel))
    ordered = pixel[order]
    first = np.r_[True, ordered[1:] != ordered[:-1]]
    return order[first]


def chroma(rgb: np.ndarray) -> np.ndarray:
    total = rgb.sum(axis=1, keepdims=True)
    return rgb / np.maximum(total, 20.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", default="004338")
    parser.add_argument("--mapped-cloud", type=Path, default=DEFAULT_CLOUD)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--yaw-step-deg", type=float, default=2.0)
    args = parser.parse_args()

    frame = next(item for item in load_frames() if item.frame_id == args.frame_id)
    with Image.open(frame.image_path) as source:
        height = round(args.width * source.height / source.width)
        panorama = np.asarray(source.resize((args.width, height), Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)

    records = read_binary_pcd(args.mapped_cloud, COLOR_DTYPE)
    world_xyz = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64)
    colors = packed_rgb(records)
    camera_from_world = world_from_flu(*frame.attitude).T
    camera_xyz = (camera_from_world @ (world_xyz - frame.position).T).T
    depth = np.linalg.norm(camera_xyz, axis=1)
    valid = np.isfinite(camera_xyz).all(axis=1) & (depth >= 2.0) & (depth <= 80.0)
    camera_xyz, depth, colors = camera_xyz[valid], depth[valid], colors[valid]
    alpha = np.arctan2(camera_xyz[:, 1], camera_xyz[:, 0])
    omega = np.arcsin(np.clip(camera_xyz[:, 2] / depth, -1.0, 1.0))
    base_v = np.floor((0.5 - omega / math.pi) * height + 0.5).astype(np.int64)
    vertical = (base_v >= 0) & (base_v < height)
    alpha, base_v, depth, colors = alpha[vertical], base_v[vertical], depth[vertical], colors[vertical]

    results: list[dict[str, object]] = []
    yaw_values = np.arange(-180.0, 180.0, args.yaw_step_deg)
    for label, horizontal_sign in (("given_formula", -1.0), ("horizontal_mirror", 1.0)):
        base_u = (0.5 + horizontal_sign * alpha / (2.0 * math.pi)) % 1.0
        # Visibility/collisions are unchanged by a circular yaw shift or mirror.
        base_pixel_u = np.floor(base_u * args.width + 0.5).astype(np.int64) % args.width
        chosen = nearest_per_pixel(base_v * args.width + base_pixel_u, depth)
        selected_u = base_u[chosen]
        selected_v = base_v[chosen]
        selected_colors = colors[chosen]
        selected_chroma = chroma(selected_colors)
        for yaw_deg in yaw_values:
            u = np.floor(((selected_u + yaw_deg / 360.0) % 1.0) * args.width + 0.5).astype(np.int64) % args.width
            image_colors = panorama[selected_v, u]
            rgb_mae = float(np.mean(np.abs(image_colors - selected_colors)))
            chroma_mae = float(np.mean(np.abs(chroma(image_colors) - selected_chroma)))
            results.append(
                {
                    "orientation": label,
                    "yaw_deg": float(yaw_deg),
                    "rgb_mae": rgb_mae,
                    "chroma_mae": chroma_mae,
                    "visible_pixels": int(len(chosen)),
                }
            )

    by_chroma = sorted(results, key=lambda item: (item["chroma_mae"], item["rgb_mae"]))
    for row in by_chroma[:12]:
        print(
            f"{row['orientation']:18s} yaw={row['yaw_deg']:7.1f} "
            f"chroma_mae={row['chroma_mae']:.6f} rgb_mae={row['rgb_mae']:.3f} "
            f"pixels={row['visible_pixels']:,}"
        )
    best_by_orientation = {}
    for label in ("given_formula", "horizontal_mirror"):
        best_by_orientation[label] = min(
            (row for row in results if row["orientation"] == label),
            key=lambda item: (item["chroma_mae"], item["rgb_mae"]),
        )
    report = {
        "frame_id": frame.frame_id,
        "mapped_cloud": str(args.mapped_cloud),
        "method": "vendor point RGB versus panorama pixel RGB after z-buffering; lower is better",
        "best_by_orientation": best_by_orientation,
        "top_results": by_chroma[:30],
    }
    output_path = OUTPUT_ROOT / f"frame_{frame.frame_id}" / "panorama_handedness_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
