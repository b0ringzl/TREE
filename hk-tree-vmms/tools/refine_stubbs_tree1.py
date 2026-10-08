"""High-purity, partial tree-1 cloud from the audited 001557 image polygon.

The broad exploratory candidate is retained separately.  This export removes
its slope/root wedge because the lower points cannot be separated reliably
from the vegetated hillside with the available single-view annotations.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

import extract_stubbs_tree1 as extract
import probe_stubbs_tree1 as common


OUT = common.OUT / "crown_polygon_v1"
SOURCE = common.OUT / "candidate_v1/candidate_tree1.pcd"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite refined cloud: {OUT}")
    OUT.mkdir(parents=True)
    scene = json.loads((common.SCENE / "场景说明.json").read_text(encoding="utf-8"))
    annotation = json.loads((common.SCENE / "影像数据/annotation.json").read_text(encoding="utf-8"))
    candidate = common.read_pcd(SOURCE)
    xyz = np.column_stack([candidate[k] for k in "xyz"])
    px, py = common.project(xyz, scene)
    keep = common.polygon_mask(annotation)[py, px] & (xyz[:, 2] >= -47.5) & (xyz[:, 2] <= -27)
    chosen = np.asarray(candidate[keep])
    coords = xyz[keep]
    voxel_size = .4
    ijk = np.floor((coords - np.array([-356, 621, -50])) / voxel_size).astype(np.int32)
    shape = tuple(np.max(ijk, axis=0) + 1)
    occupied = np.zeros(shape, dtype=bool)
    occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    labels, _ = ndimage.label(occupied, structure=np.ones((3, 3, 3), dtype=np.uint8))
    component = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    counts = np.bincount(component)
    largest = int(np.argmax(counts[1:]) + 1)
    chosen = chosen[component == largest]
    coords = coords[component == largest].astype(np.float32)
    sha = extract.write_pcd(OUT / "tree1_partial_clean.pcd", chosen)
    np.savez_compressed(OUT / "tree1_demo_upload.npz", xyz=coords)
    extract.previews(chosen, scene, common.SCENE / "影像数据/panorama_original.jpg", OUT)
    q = np.quantile(coords, [.005, .5, .995], axis=0)
    unique_20cm = len(np.unique(np.floor(coords / .2).astype(np.int32), axis=0))
    report = {
        "status": "partial_tree_candidate_requires_visual_review",
        "species_label_from_human_image_annotation": annotation["labels"][0]["species"],
        "label_id": annotation["labels"][0]["label_id"],
        "source_scene": str(common.PCD),
        "source_candidate": str(SOURCE),
        "selection": "exact human tree-1 canopy polygon, map XYZ ROI from candidate, z in [-47.5,-27] m, largest 0.4 m 26-connected voxel component",
        "reason_lower_parts_omitted": "Visible aerial roots and hillside overlap in single-view projection; candidate_v1 lower wedge is contaminated by slope returns.",
        "points": int(len(coords)),
        "independent_20cm_voxels": unique_20cm,
        "robust_99pct_extent_m": (q[2] - q[0]).tolist(),
        "min_xyz": coords.min(axis=0).tolist(),
        "max_xyz": coords.max(axis=0).tolist(),
        "pcd_sha256": sha,
        "source_file_ids": {str(int(k)): int(v) for k, v in zip(*np.unique(chosen["source_file_id"], return_counts=True))},
        "limitations": ["upper crown and some branch structure only", "base/root not confirmed", "no claim of complete isolated tree", "image polygon and point cloud alignment based on empirical yaw correction"],
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("points", "independent_20cm_voxels", "robust_99pct_extent_m", "source_file_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
