"""Auditable tree-1 candidate from the full-density Stubbs Road 001557 scene.

This is an instance *candidate*, not a certified complete single tree.  The
original PCD is read only.  Image polygons, a stem wedge, and a map-coordinate
ROI restrict the exported points; point source IDs are retained.  No species
model or trained segmentation model is involved.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

import probe_stubbs_tree1 as common


OUT = common.OUT / "candidate_v1"
DTYPE = common.DTYPE
X_RANGE = (-356.0, -336.0)
Y_RANGE = (621.0, 641.0)
Z_RANGE = (-50.0, -27.0)
VOXEL_M = 0.4


def image_masks(annotation: dict) -> tuple[np.ndarray, np.ndarray]:
    canopy = Image.fromarray(common.polygon_mask(annotation).astype(np.uint8) * 255, "L")
    canopy = np.asarray(canopy.filter(ImageFilter.MaxFilter(35))) > 0
    stem = Image.new("1", (2048, 1024))
    # Approximate visible stem/aerial-root wedge beneath the human canopy mask.
    ImageDraw.Draw(stem).polygon(
        [(1310, 185), (1510, 175), (1650, 230), (1680, 400),
         (1590, 555), (1375, 590), (1280, 390)],
        fill=1,
    )
    return canopy, np.asarray(stem, dtype=bool)


def write_pcd(path: Path, records: np.ndarray) -> str:
    if path.exists():
        raise FileExistsError(path)
    header = (
        "# .PCD v0.7\nVERSION 0.7\n"
        "FIELDS x y z intensity rgb source_file_id source_point_id\n"
        "SIZE 4 4 4 4 4 4 4\nTYPE F F F F U U U\n"
        "COUNT 1 1 1 1 1 1 1\n"
        f"WIDTH {len(records)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(records)}\nDATA binary\n"
    ).encode("ascii")
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        stream.write(header)
        digest.update(header)
        for start in range(0, len(records), 500_000):
            payload = records[start:start + 500_000].tobytes()
            stream.write(payload)
            digest.update(payload)
    assert path.stat().st_size == len(header) + len(records) * DTYPE.itemsize
    return digest.hexdigest()


def components(xyz: np.ndarray) -> list[dict]:
    voxel = np.floor((xyz - np.array([X_RANGE[0], Y_RANGE[0], Z_RANGE[0]])) / VOXEL_M).astype(np.int32)
    shape = tuple(np.max(voxel, axis=0) + 1)
    occupied = np.zeros(shape, dtype=bool)
    occupied[voxel[:, 0], voxel[:, 1], voxel[:, 2]] = True
    labels, number = ndimage.label(occupied, structure=np.ones((3, 3, 3), dtype=np.uint8))
    point_labels = labels[voxel[:, 0], voxel[:, 1], voxel[:, 2]]
    counts = np.bincount(point_labels, minlength=number + 1)
    order = np.argsort(counts[1:])[::-1][:12] + 1
    rows = []
    for label in order:
        subset = xyz[point_labels == label]
        rows.append({
            "component_id": int(label), "points": int(len(subset)),
            "min_xyz": subset.min(axis=0).tolist(),
            "max_xyz": subset.max(axis=0).tolist(),
        })
    return rows


def previews(records: np.ndarray, scene: dict, panorama: Path, out: Path = OUT) -> None:
    xyz = np.column_stack([records[k] for k in "xyz"])
    xyz = xyz[::max(1, len(xyz) // 100_000)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, a, b, title in [
        (axes[0], 0, 1, "Top / XY"),
        (axes[1], 0, 2, "Side / XZ"),
        (axes[2], 1, 2, "Side / YZ"),
    ]:
        ax.scatter(xyz[:, a], xyz[:, b], c=xyz[:, 2], s=.07, alpha=.32, cmap="turbo", rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("XYZ"[a] + " (m)")
        ax.set_ylabel("XYZ"[b] + " (m)")
        ax.set_aspect("equal")
    fig.savefig(out / "candidate_views.png", dpi=150)
    plt.close(fig)
    px, py = common.project(xyz, scene)
    base = Image.open(panorama).convert("RGB").resize((2048, 1024))
    layer = np.zeros((1024, 2048, 4), dtype=np.uint8)
    layer[py, px] = [255, 210, 20, 230]
    marked = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(layer, "RGBA"))
    marked.convert("RGB").save(out / "candidate_projection.jpg", quality=93)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite candidate: {OUT}")
    OUT.mkdir(parents=True)
    scene = json.loads((common.SCENE / "场景说明.json").read_text(encoding="utf-8"))
    annotation = json.loads((common.SCENE / "影像数据/annotation.json").read_text(encoding="utf-8"))
    canopy, stem = image_masks(annotation)
    cloud = common.read_pcd()
    parts = []
    stats = {"spatial": 0, "canopy": 0, "stem": 0}
    for start in range(0, len(cloud), 1_000_000):
        records = cloud[start:start + 1_000_000]
        xyz = np.column_stack([records[k] for k in "xyz"])
        spatial = (
            np.isfinite(xyz).all(axis=1)
            & (xyz[:, 0] >= X_RANGE[0]) & (xyz[:, 0] <= X_RANGE[1])
            & (xyz[:, 1] >= Y_RANGE[0]) & (xyz[:, 1] <= Y_RANGE[1])
            & (xyz[:, 2] >= Z_RANGE[0]) & (xyz[:, 2] <= Z_RANGE[1])
        )
        if not spatial.any():
            continue
        stats["spatial"] += int(spatial.sum())
        chosen = records[spatial]
        xyz = xyz[spatial]
        px, py = common.project(xyz, scene)
        upper = canopy[py, px] & (xyz[:, 2] >= -47.8)
        lower = stem[py, px] & (xyz[:, 2] <= -34.0)
        stats["canopy"] += int(upper.sum())
        stats["stem"] += int(lower.sum())
        keep = upper | lower
        if keep.any():
            parts.append(np.array(chosen[keep]))
        if start % 10_000_000 == 0:
            print("scanned", min(start + len(records), len(cloud)), "candidate", sum(len(p) for p in parts), flush=True)
    candidate = np.concatenate(parts)
    xyz = np.column_stack([candidate[k] for k in "xyz"])
    report = {
        "source_scene": str(common.PCD),
        "source_scene_sha256": scene["pcd_sha256"],
        "annotation": str(common.SCENE / "影像数据/annotation.json"),
        "label_id": annotation["labels"][0]["label_id"],
        "selection": {"xy_roi": [X_RANGE, Y_RANGE], "z_roi": Z_RANGE,
                      "canopy": "human tree-1 polygon dilated 17 panorama pixels, z >= -47.8",
                      "stem": "documented image wedge, z <= -34.0",
                      "source_records_kept": True},
        "statistics": stats,
        "candidate_points": int(len(candidate)),
        "candidate_min_xyz": xyz.min(axis=0).tolist(),
        "candidate_max_xyz": xyz.max(axis=0).tolist(),
        "components_0_4m_26connected": components(xyz),
    }
    report["candidate_pcd_sha256"] = write_pcd(OUT / "candidate_tree1.pcd", candidate)
    previews(candidate, scene, common.SCENE / "影像数据/panorama_original.jpg")
    (OUT / "candidate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("candidate_points", len(candidate), "components", report["components_0_4m_26connected"][:4], flush=True)


if __name__ == "__main__":
    main()
