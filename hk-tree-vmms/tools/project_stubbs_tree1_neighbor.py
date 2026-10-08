"""Independent visual pairing check: project cleaned 001557 points onto 001562."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import probe_stubbs_tree1 as common


sys.path.insert(0, str(common.ROOT / "hk-tree-vmms"))
from tools import build_selected_raw_multimodal_scenes as shared  # noqa: E402


OUT = common.OUT / "neighbor_projection_v1"
CLOUD = common.OUT / "visible_depth_1m_v1/tree1_visible_partial.pcd"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True)
    row = shared.read_coordinates()["stubbs_pano_1__001562"]
    scene = {
        "camera_position_map_xyz": [float(row[k]) for k in ("local_x", "local_y", "local_z")],
        "world_from_camera": (
            shared.world_from_flu(*[float(row[f"{k}_rad"]) for k in ("roll", "pitch", "heading")])
            @ np.diag([-1., -1., 1.])
        ).tolist(),
    }
    cloud = common.read_pcd(CLOUD)
    stride = max(1, len(cloud) // 160_000)
    xyz = np.column_stack([cloud[k][::stride] for k in "xyz"])
    px, py = common.project(xyz, scene)
    source = shared.VMMS / row["source_image_relpath"]
    base = Image.open(source).convert("RGB").resize((2048, 1024))
    layer = np.zeros((1024, 2048, 4), dtype=np.uint8)
    layer[py, px] = [255, 208, 0, 230]
    marked = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(layer, "RGBA"))
    marked.convert("RGB").save(OUT / "tree1_points_on_001562_panorama.jpg", quality=93)
    report = {
        "source_cloud": str(CLOUD), "source_frame": str(source),
        "frame_id": row["frame_id"], "sampled_points_projected": int(len(xyz)),
        "projection": "same empirical-yaw INS projection as source scene; visual QA only, not precision extrinsic calibration",
        "image_coverage_bbox_px_2048x1024": [int(px.min()), int(py.min()), int(px.max()), int(py.max())],
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
