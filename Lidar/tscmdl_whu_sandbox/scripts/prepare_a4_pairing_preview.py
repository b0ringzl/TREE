"""Create coarse point-cloud/panorama pairing previews for selected A3 samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    minimal_circular_interval,
    nearest_poses,
    orientation_column_diagnostics,
    project_equirectangular,
    read_trajectory_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--road", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views-per-tree", type=int, default=2)
    return parser.parse_args()


def wrapped_crop(image: Image.Image, left: float, top: float, width: float, height: float) -> Image.Image:
    image_width, image_height = image.size
    crop_width = max(1, min(int(round(width)), image_width))
    crop_height = max(1, min(int(round(height)), image_height))
    top_int = max(0, min(int(round(top)), image_height - crop_height))
    left_mod = int(round(left)) % image_width
    if left_mod + crop_width <= image_width:
        return image.crop((left_mod, top_int, left_mod + crop_width, top_int + crop_height))
    first_width = image_width - left_mod
    output = Image.new(image.mode, (crop_width, crop_height))
    output.paste(image.crop((left_mod, top_int, image_width, top_int + crop_height)), (0, 0))
    output.paste(
        image.crop((0, top_int, crop_width - first_width, top_int + crop_height)),
        (first_width, 0),
    )
    return output


def draw_overlay(
    crop: Image.Image,
    u: np.ndarray,
    v: np.ndarray,
    left: float,
    top: float,
    panorama_width: int,
    crop_width: float,
    crop_height: float,
    label: str,
) -> tuple[Image.Image, int]:
    output = crop.resize((768, 512), Image.Resampling.LANCZOS).convert("RGBA")
    layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x = np.mod(u - left, panorama_width)
    y = v - top
    visible = (x >= 0) & (x < crop_width) & (y >= 0) & (y < crop_height)
    x_scaled = x[visible] * 768.0 / crop_width
    y_scaled = y[visible] * 512.0 / crop_height
    for px, py in zip(x_scaled[::2], y_scaled[::2]):
        draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=(255, 35, 20, 175))
    draw.rectangle((8, 8, min(760, 18 + len(label) * 7), 34), fill=(0, 0, 0, 180))
    draw.text((14, 13), label, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return Image.alpha_composite(output, layer).convert("RGB"), int(np.count_nonzero(visible))


def draw_panorama_preview(
    image: Image.Image, u: np.ndarray, v: np.ndarray, label: str
) -> Image.Image:
    preview = image.resize((1600, 800), Image.Resampling.LANCZOS).convert("RGBA")
    layer = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x = u * 1600.0 / image.width
    y = v * 800.0 / image.height
    visible = (y >= 0) & (y < 800)
    for px, py in zip(x[visible][::4], y[visible][::4]):
        draw.ellipse((px - 1.2, py - 1.2, px + 1.2, py + 1.2), fill=(255, 35, 20, 180))
    draw.rectangle((8, 8, min(1590, 18 + len(label) * 7), 34), fill=(0, 0, 0, 180))
    draw.text((14, 13), label, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return Image.alpha_composite(preview, layer).convert("RGB")


def main() -> None:
    args = parse_args()
    if args.views_per_tree <= 0:
        raise ValueError("views-per-tree must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_csv = args.dataset_root / args.road / "hdi" / args.trajectory / "traj.csv"
    image_dir = args.dataset_root / args.road / "image" / args.trajectory
    poses = read_trajectory_csv(trajectory_csv)
    diagnostics = orientation_column_diagnostics(trajectory_csv)
    if diagnostics["heading_column"] != 3:
        raise ValueError(f"Expected raw orientation column 3 to be heading: {diagnostics}")

    outputs = []
    for sample_path in sorted(args.sample_dir.glob("*.npz")):
        with np.load(sample_path, allow_pickle=False) as archive:
            points = archive["points_xyz_raw"]
            tree_id = int(archive["tree_id"])
            raw_label_id = int(archive["raw_label_id"])
            benchmark_label_id = int(archive["benchmark_label_id"])
        center = points.astype(np.float64).mean(axis=0)

        for view_rank, (pose, center_distance) in enumerate(
            nearest_poses(center, poses, count=args.views_per_tree), start=1
        ):
            image_path = image_dir / pose.image_name
            with Image.open(image_path) as source:
                panorama = source.convert("RGB")
            u, v, distances = project_equirectangular(
                points, pose, panorama.width, panorama.height, apply_tilt=True
            )
            arc_start, arc_width = minimal_circular_interval(u, panorama.width)
            vertical_low, vertical_high = np.percentile(v, [0.5, 99.5])

            crop_width = min(
                panorama.width,
                max(256.0, arc_width * 1.25),
            )
            crop_height = min(
                panorama.height,
                max(256.0, (vertical_high - vertical_low) * 1.25),
            )
            crop_left = arc_start + arc_width / 2.0 - crop_width / 2.0
            crop_top = (vertical_low + vertical_high) / 2.0 - crop_height / 2.0
            crop_top = max(0.0, min(crop_top, panorama.height - crop_height))

            raw_crop = wrapped_crop(
                panorama, crop_left, crop_top, crop_width, crop_height
            ).resize((768, 512), Image.Resampling.LANCZOS)
            label = (
                f"tree {tree_id} | raw {raw_label_id} | class {benchmark_label_id} | "
                f"view {view_rank} | {center_distance:.1f} m"
            )
            overlay, visible_point_count = draw_overlay(
                wrapped_crop(panorama, crop_left, crop_top, crop_width, crop_height),
                u,
                v,
                crop_left,
                crop_top,
                panorama.width,
                crop_width,
                crop_height,
                label,
            )
            panorama_preview = draw_panorama_preview(panorama, u, v, label)

            stem = f"tree_{tree_id}_view_{view_rank}"
            raw_path = args.output_dir / f"{stem}_raw.jpg"
            overlay_path = args.output_dir / f"{stem}_overlay.jpg"
            panorama_path = args.output_dir / f"{stem}_panorama.jpg"
            raw_crop.save(raw_path, quality=92)
            overlay.save(overlay_path, quality=92)
            panorama_preview.save(panorama_path, quality=90)
            outputs.append(
                {
                    "tree_id": tree_id,
                    "raw_label_id": raw_label_id,
                    "benchmark_label_id": benchmark_label_id,
                    "view_rank": view_rank,
                    "image_name": pose.image_name,
                    "pose": pose.to_dict(),
                    "tree_center": center.tolist(),
                    "center_distance_m": center_distance,
                    "point_distance_range_m": [float(distances.min()), float(distances.max())],
                    "arc_start_px": arc_start,
                    "arc_width_px": arc_width,
                    "crop_left_unwrapped_px": crop_left,
                    "crop_top_px": crop_top,
                    "crop_width_px": crop_width,
                    "crop_height_px": crop_height,
                    "visible_projected_point_count": visible_point_count,
                    "raw_crop_path": str(raw_path),
                    "overlay_path": str(overlay_path),
                    "panorama_preview_path": str(panorama_path),
                }
            )

    manifest = {
        "road_id": args.road,
        "trajectory_id": args.trajectory,
        "projection_status": (
            "coarse prototype; camera lever arm and exact extrinsics are not present "
            "in the local public dataset package"
        ),
        "coordinate_convention": "local X right/east, Y forward/north, Z up; heading clockwise from +Y",
        "orientation_order": "roll, pitch, heading inferred from trajectory bearing",
        "panorama_model": "full equirectangular, center column is forward",
        "orientation_diagnostics": diagnostics,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "a4_pairing_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
