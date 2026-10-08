"""Read-only spatial probe for the Stubbs Road 001557 / 001562 banyan scene.

The exported PCD is memory-mapped and sampled only for diagnostics.  No source
file is modified, and this program does not assert that projected canopy points
are a clean single-tree instance.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "交付/人工标注_原始LiDAR配准场景_20260907_v4_不降采样/05_SB_stubbs_1_001557_榕树"
OUT = ROOT / "outputs/stubbs_tree1_clean_20260922_v1"
PCD = SCENE / "点云数据/lidar_scene_full_density.pcd"
DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("intensity", "<f4"), ("rgb", "<u4"),
    ("source_file_id", "<u4"), ("source_point_id", "<u4"),
])


def read_pcd(path: Path = PCD) -> np.memmap:
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if line.startswith(b"POINTS "):
                count = int(line.split()[1])
            if line.startswith(b"DATA "):
                assert line == b"DATA binary\n"
                offset = stream.tell()
                break
    assert path.stat().st_size == offset + count * DTYPE.itemsize
    return np.memmap(path, dtype=DTYPE, mode="r", offset=offset, shape=(count,))


def polygon_mask(annotation: dict, index: int = 0) -> np.ndarray:
    mask = Image.new("1", (2048, 1024))
    points = [(round(x * 2048), round(y * 1024)) for x, y in annotation["labels"][index]["points"]]
    ImageDraw.Draw(mask).polygon(points, fill=1)
    return np.asarray(mask, dtype=bool)


def project(xyz: np.ndarray, scene: dict) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(scene["camera_position_map_xyz"], dtype=np.float64)
    rotation = np.asarray(scene["world_from_camera"], dtype=np.float64)
    cam = (xyz.astype(np.float64) - center) @ rotation
    distance = np.linalg.norm(cam, axis=1)
    u = (0.5 - np.arctan2(cam[:, 1], cam[:, 0]) / (2 * np.pi)) % 1
    v = 0.5 - np.arcsin(np.clip(cam[:, 2] / np.maximum(distance, 1e-6), -1, 1)) / np.pi
    return np.minimum((u * 2048).astype(np.int32), 2047), np.clip((v * 1024).astype(np.int32), 0, 1023)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = json.loads((SCENE / "场景说明.json").read_text(encoding="utf-8"))
    annotation = json.loads((SCENE / "影像数据/annotation.json").read_text(encoding="utf-8"))
    cloud = read_pcd()
    step = 20
    subset = cloud[::step]
    xyz = np.column_stack([subset[k] for k in "xyz"])
    # HK80 easting/northing minus the fixed vendor-map origin for this survey.
    tree_xy = np.array([836292.174 - 836624.967, 815027.498 - 814410.225], dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    radius = np.linalg.norm(xyz[:, :2] - tree_xy, axis=1)
    roi = finite & (radius <= 23)
    px, py = project(xyz[roi], scene)
    poly = polygon_mask(annotation)[py, px]
    # Diagnostic only: the hand-drawn polygon covers the canopy but excludes
    # its descending stems.  Probe their visible angular wedge separately.
    stem_wedge = (px >= 1350) & (px <= 1610) & (py >= 205) & (py <= 590)
    broad_wedge = (px >= 1130) & (px <= 1920) & (py >= 0) & (py <= 610)
    selected = xyz[roi]
    z = selected[:, 2]
    poly_z = z[poly]
    print("source", len(cloud), "sample", len(subset), "roi", len(selected), "polygon", int(poly.sum()))
    print("tree_xy", tree_xy.tolist())
    print("roi_z_quantiles", np.quantile(z, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist())
    print("poly_z_quantiles", np.quantile(poly_z, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist())
    extent = [float(tree_xy[0] - 23), float(tree_xy[0] + 23), float(tree_xy[1] - 23), float(tree_xy[1] + 23)]
    bins = [np.linspace(extent[0], extent[1], 185), np.linspace(extent[2], extent[3], 185)]
    strata = [
        ("All points", np.ones(len(selected), dtype=bool)),
        ("Lower: z < -48", z < -48),
        ("Mid: -48 to -40", (z >= -48) & (z < -40)),
        ("Upper: z >= -40", z >= -40),
        ("Tree 1 image polygon", poly),
        ("Polygon & upper", poly & (z >= -40)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    for ax, (name, choose) in zip(axes.flat, strata):
        h, _, _ = np.histogram2d(selected[choose, 0], selected[choose, 1], bins=bins)
        ax.imshow(np.log1p(h.T), extent=extent, origin="lower", cmap="viridis", aspect="equal")
        ax.plot(tree_xy[0], tree_xy[1], "r+", markersize=16, markeredgewidth=2)
        ax.set_title(f"{name} ({int(choose.sum()):,} / 20)")
        ax.set_xlabel("Vendor map X (m)")
        ax.set_ylabel("Vendor map Y (m)")
    fig.savefig(OUT / "probe_xy.png", dpi=140)
    plt.close(fig)
    detail_extent = [-350, -326, 616, 640]
    detail_bins = [np.linspace(detail_extent[0], detail_extent[1], 193), np.linspace(detail_extent[2], detail_extent[3], 193)]
    detail = [
        ("Canopy polygon", poly),
        ("Stem image wedge", stem_wedge),
        ("Stem wedge z -49..-43", stem_wedge & (z >= -49) & (z < -43)),
        ("Stem wedge z >= -43", stem_wedge & (z >= -43)),
        ("Broad wedge z -49..-43", broad_wedge & (z >= -49) & (z < -43)),
        ("Broad wedge z >= -43", broad_wedge & (z >= -43)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    for ax, (name, choose) in zip(axes.flat, detail):
        h, _, _ = np.histogram2d(selected[choose, 0], selected[choose, 1], bins=detail_bins)
        ax.imshow(np.log1p(h.T), extent=detail_extent, origin="lower", cmap="viridis", aspect="equal")
        ax.plot(tree_xy[0], tree_xy[1], "r+", markersize=16, markeredgewidth=2)
        ax.set_title(f"{name} ({int(choose.sum()):,} / 20)")
        ax.set_xlabel("Vendor map X (m)")
        ax.set_ylabel("Vendor map Y (m)")
    fig.savefig(OUT / "probe_detail_xy.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        instance_mask = polygon_mask(annotation, index)[py, px]
        h, _, _ = np.histogram2d(selected[instance_mask, 0], selected[instance_mask, 1], bins=bins)
        ax.imshow(np.log1p(h.T), extent=extent, origin="lower", cmap="viridis", aspect="equal")
        ax.set_title(f"Annotated tree {index+1}: {int(instance_mask.sum()):,} / 20")
        ax.set_xlabel("Vendor map X (m)")
        ax.set_ylabel("Vendor map Y (m)")
    fig.savefig(OUT / "probe_three_annotations_xy.png", dpi=140)
    plt.close(fig)
    focus_extent = [-356, -334, 620, 643]
    focus_bins = [np.linspace(focus_extent[0], focus_extent[1], 89), np.linspace(focus_extent[2], focus_extent[3], 93)]
    ix = np.searchsorted(focus_bins[0], selected[:, 0], side="right") - 1
    iy = np.searchsorted(focus_bins[1], selected[:, 1], side="right") - 1
    inside = (ix >= 0) & (ix < 88) & (iy >= 0) & (iy < 92)
    flat = ix[inside] * 92 + iy[inside]
    min_z = np.full(88 * 92, np.inf, dtype=np.float32)
    counts = np.zeros(88 * 92, dtype=np.int32)
    np.minimum.at(min_z, flat, selected[inside, 2])
    np.add.at(counts, flat, 1)
    min_z[counts < 4] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].imshow(min_z.reshape(88, 92).T, extent=focus_extent, origin="lower", cmap="terrain", vmin=-53, vmax=-30)
    axes[0].set_title("Lowest sampled Z per 0.25 m cell")
    axes[1].imshow(np.log1p(counts.reshape(88, 92).T), extent=focus_extent, origin="lower", cmap="viridis")
    axes[1].set_title("Point density per cell (1/20 sample)")
    for ax in axes:
        ax.set_xlabel("Vendor map X (m)")
        ax.set_ylabel("Vendor map Y (m)")
    fig.savefig(OUT / "probe_ground.png", dpi=140)
    plt.close(fig)
    for cx, cy in [(-343, 628), (-343, 629), (-344, 628), (-341, 629)]:
        near = np.linalg.norm(selected[:, :2] - np.array([cx, cy]), axis=1) < 0.8
        if near.sum():
            print("column", cx, cy, int(near.sum()), np.quantile(z[near], [0, .05, .25, .5, .75, .95, 1]).tolist())
    report = {
        "source_pcd": str(PCD), "source_points": len(cloud), "sample_step": step,
        "tree_xy": tree_xy.tolist(), "roi_sample_points": len(selected),
        "tree1_polygon_sample_points": int(poly.sum()),
        "roi_z_quantiles": np.quantile(z, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist(),
        "polygon_z_quantiles": np.quantile(poly_z, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist(),
    }
    (OUT / "probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
