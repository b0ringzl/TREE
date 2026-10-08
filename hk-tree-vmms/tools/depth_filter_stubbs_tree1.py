"""Visibility-filter the Stubbs tree-1 crown candidate without altering sources."""
from __future__ import annotations

import json

import numpy as np

import extract_stubbs_tree1 as extract
import probe_stubbs_tree1 as common


SOURCE = common.OUT / "crown_polygon_v1/tree1_partial_clean.pcd"


def main() -> None:
    scene = json.loads((common.SCENE / "场景说明.json").read_text(encoding="utf-8"))
    source = common.read_pcd(SOURCE)
    xyz = np.column_stack([source[k] for k in "xyz"])
    px, py = common.project(xyz, scene)
    pixel = py * 2048 + px
    camera = np.asarray(scene["camera_position_map_xyz"], dtype=np.float64)
    distance = np.linalg.norm(xyz.astype(np.float64) - camera, axis=1).astype(np.float32)
    nearest = np.full(2048 * 1024, np.inf, dtype=np.float32)
    np.minimum.at(nearest, pixel, distance)
    delta = distance - nearest[pixel]
    for metres in (1.0, 2.0):
        out = common.OUT / f"visible_depth_{int(metres)}m_v1"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite: {out}")
        out.mkdir(parents=True)
        keep = delta <= metres + 1e-5
        chosen = np.asarray(source[keep])
        coords = xyz[keep].astype(np.float32)
        sha = extract.write_pcd(out / "tree1_visible_partial.pcd", chosen)
        np.savez_compressed(out / "tree1_demo_upload.npz", xyz=coords)
        extract.previews(chosen, scene, common.SCENE / "影像数据/panorama_original.jpg", out)
        q = np.quantile(coords, [.005, .5, .995], axis=0)
        report = {
            "status": "partial_tree_candidate_requires_visual_review",
            "source": str(SOURCE),
            "algorithm": "Per-pixel first visible depth in 2048x1024 equirectangular frame; retain points within tolerance. This is not a calibrated occlusion or tree-instance segmentation.",
            "depth_tolerance_m": metres,
            "source_points": int(len(source)),
            "kept_points": int(len(chosen)),
            "robust_99pct_extent_m": (q[2] - q[0]).tolist(),
            "independent_20cm_voxels": int(len(np.unique(np.floor(coords / .2).astype(np.int32), axis=0))),
            "pcd_sha256": sha,
            "limitations": ["crown and branches only", "base/root omitted", "same-pixel closest depth can select foreground noise", "empirical yaw correction, no precision extrinsics"],
        }
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(out.name, report["kept_points"], report["independent_20cm_voxels"], report["robust_99pct_extent_m"], flush=True)


if __name__ == "__main__":
    main()
