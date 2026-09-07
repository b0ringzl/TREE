from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent
TREE_ROOT = PROJECT_ROOT.parent
VMMS_ROOT = TREE_ROOT.parent / "vmms"
FRAME_CSV = PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv"
VIEW_CACHE = PROJECT_ROOT / "derived" / "hewentian_labeler" / "view_cache"
OUTPUT_ROOT = PROJECT_ROOT / "derived" / "pointcloud_preview"
COLOR_CLOUD_DIR = (
    VMMS_ROOT / "2023-10-27_hewentian" / "pointcloud" / "ColorCloudPoint"
)
GEOREFERENCE_MATRIX = (
    VMMS_ROOT
    / "2023-10-27_hewentian"
    / "pointcloud"
    / "GeoReference_Transformation_Matrix.txt"
)


def signed_angle_deg(value: np.ndarray | float) -> np.ndarray | float:
    return (value + 180.0) % 360.0 - 180.0


def parse_hex_color(value: str) -> tuple[int, int, int]:
    normalized = value.strip().removeprefix("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", normalized):
        raise argparse.ArgumentTypeError("Color must use RRGGBB or #RRGGBB format")
    return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))


def load_origin() -> tuple[float, float, float]:
    rows = [
        [float(value) for value in line.split()]
        for line in GEOREFERENCE_MATRIX.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[0][3], rows[1][3], rows[2][3]


def load_frame(frame_id: str) -> dict[str, str]:
    with FRAME_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] == "hewentian_pano" and row["frame_id"] == frame_id:
                return row
    raise RuntimeError(f"Frame {frame_id} was not found in {FRAME_CSV}")


def color_clouds_in_window(
    frame_id: str, window_frames: int
) -> list[tuple[Path, int]]:
    frame_number = int(frame_id)
    candidates: list[tuple[int, int, Path]] = []
    pattern = re.compile(r"_img(?P<image>\d+)_")
    for path in COLOR_CLOUD_DIR.glob("*.pcd"):
        match = pattern.search(path.name)
        if match:
            image_number = int(match.group("image"))
            candidates.append((abs(image_number - frame_number), image_number, path))
    if not candidates:
        raise RuntimeError(f"No colorized PCD files found in {COLOR_CLOUD_DIR}")
    candidates.sort(key=lambda item: (item[0], item[1]))
    if window_frames <= 0:
        _, image_number, path = candidates[0]
        return [(path, image_number)]
    selected = [
        (path, image_number)
        for delta, image_number, path in candidates
        if delta <= window_frames
    ]
    return selected or [(candidates[0][2], candidates[0][1])]


def read_binary_xyzi_rgb(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines: list[bytes] = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Incomplete PCD header: {path}")
            header_lines.append(line)
            if line.startswith(b"DATA "):
                break
        header = b"".join(header_lines).decode("ascii", errors="replace")
        if "DATA binary" not in header:
            raise RuntimeError(f"Only binary PCD is supported: {path}")
        if "FIELDS x y z rgb" not in header:
            raise RuntimeError(f"Unexpected PCD schema: {path}")
        point_match = re.search(r"^POINTS\s+(\d+)\s*$", header, flags=re.MULTILINE)
        if not point_match:
            raise RuntimeError(f"POINTS field is missing: {path}")
        point_count = int(point_match.group(1))
        payload = handle.read(point_count * 16)
    if len(payload) != point_count * 16:
        raise RuntimeError(f"Truncated PCD payload: {path}")
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]
    )
    return np.frombuffer(payload, dtype=dtype)


def draw_points(
    width: int,
    height: int,
    u: np.ndarray,
    v: np.ndarray,
    colors: np.ndarray,
    radius: int,
    alpha: int,
) -> Image.Image:
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for x, y, color in zip(u.tolist(), v.tolist(), colors.tolist()):
        r, g, b = (int(color[0]), int(color[1]), int(color[2]))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(r, g, b, alpha),
        )
    return layer


def write_las_12_rgb(
    path: Path,
    xyz: np.ndarray,
    colors_8bit: np.ndarray,
    classifications: np.ndarray,
) -> None:
    """Write an uncompressed LAS 1.2 file using point format 2 (XYZ + RGB)."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if colors_8bit.shape != xyz.shape:
        raise ValueError("colors_8bit must have shape (N, 3)")
    if classifications.shape != (len(xyz),):
        raise ValueError("classifications must have shape (N,)")
    if not len(xyz):
        raise ValueError("Cannot write an empty LAS file")

    path.parent.mkdir(parents=True, exist_ok=True)
    scale = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    minima = xyz.min(axis=0)
    maxima = xyz.max(axis=0)
    offsets = np.floor(minima / 100.0) * 100.0
    integer_xyz = np.rint((xyz - offsets) / scale).astype(np.int32)
    point_count = len(xyz)

    header = bytearray(227)
    header[0:4] = b"LASF"
    struct.pack_into("<H", header, 4, 0)  # File source ID
    struct.pack_into("<H", header, 6, 0)  # Global encoding
    header[24] = 1
    header[25] = 2
    header[26:58] = b"HK-Tree VMMS".ljust(32, b"\0")
    header[58:90] = b"Codex LAS exporter".ljust(32, b"\0")
    now = datetime.now(timezone.utc)
    struct.pack_into("<H", header, 90, int(now.strftime("%j")))
    struct.pack_into("<H", header, 92, now.year)
    struct.pack_into("<H", header, 94, 227)
    struct.pack_into("<I", header, 96, 227)
    struct.pack_into("<I", header, 100, 0)
    struct.pack_into("<B", header, 104, 2)
    struct.pack_into("<H", header, 105, 26)
    struct.pack_into("<I", header, 107, point_count)
    struct.pack_into("<I", header, 111, point_count)
    for index in range(1, 5):
        struct.pack_into("<I", header, 111 + index * 4, 0)
    struct.pack_into("<ddd", header, 131, *scale.tolist())
    struct.pack_into("<ddd", header, 155, *offsets.tolist())
    struct.pack_into(
        "<dddddd",
        header,
        179,
        maxima[0],
        minima[0],
        maxima[1],
        minima[1],
        maxima[2],
        minima[2],
    )

    record_dtype = np.dtype(
        [
            ("x", "<i4"),
            ("y", "<i4"),
            ("z", "<i4"),
            ("intensity", "<u2"),
            ("return_flags", "u1"),
            ("classification", "u1"),
            ("scan_angle_rank", "i1"),
            ("user_data", "u1"),
            ("point_source_id", "<u2"),
            ("red", "<u2"),
            ("green", "<u2"),
            ("blue", "<u2"),
        ]
    )
    records = np.zeros(point_count, dtype=record_dtype)
    records["x"], records["y"], records["z"] = integer_xyz.T
    records["return_flags"] = 9  # Return 1 of 1.
    records["classification"] = classifications
    colors_16bit = colors_8bit.astype(np.uint16) * 257
    records["red"], records["green"], records["blue"] = colors_16bit.T
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(records.tobytes(order="C"))


def encode_data_url(path: Path, mime: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def write_inline_preview(report: dict[str, object], html_path: Path) -> None:
    outputs = report["outputs"]
    background = encode_data_url(Path(outputs["background"]), "image/jpeg")
    point_layer = encode_data_url(Path(outputs["point_layer"]), "image/png")
    target_layer = encode_data_url(Path(outputs["target_layer"]), "image/png")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fragment = f"""<div id="pc-pano-preview">
  <h2>何文田点云—全景影像近似叠加</h2>
  <div class="viz-controls">
    <label class="form-label" for="pc-opacity">点云不透明度 <output id="pc-opacity-value">70%</output>
      <input class="form-range" id="pc-opacity" type="range" min="0" max="100" value="70">
    </label>
    <label class="form-check">
      <input class="form-check-input" id="pc-context-toggle" type="checkbox" checked>
      <span class="form-check-label">场景点云（橙色）</span>
    </label>
    <label class="form-check">
      <input class="form-check-input" id="pc-target-toggle" type="checkbox" checked>
      <span class="form-check-label">目标树 {report['target_radius_m']:g} m 候选点</span>
    </label>
  </div>
  <div class="pc-stage" role="img" aria-label="何文田 {report['tree_id']} 全景定向裁剪与彩色点云近似叠加；青色为树点坐标 {report['target_radius_m']:g} 米范围内的候选点，黄色竖线为树木清单坐标方位。">
    <img class="pc-background" src="{background}" alt="何文田全景影像定向裁剪">
    <img class="pc-layer" id="pc-context" src="{point_layer}" alt="彩色场景点云投影层">
    <img class="pc-layer" id="pc-target" src="{target_layer}" alt="目标树候选点云投影层">
  </div>
  <div class="viz-row text-small text-muted" aria-live="polite">
    <span>树号 {report['tree_id']}</span>
    <span>影像帧 {report['frame_id']}</span>
    <span>合并点云 {report['merged_pointcloud_file_count']} 份</span>
    <span>可投影点 {report['visible_projected_point_count']:,}</span>
    <span>{report['target_radius_m']:g} m 候选点 {report['visible_target_point_count']:,}</span>
    <span>无 LiDAR–Ladybug 完整外参</span>
  </div>
</div>
<style>
  #pc-pano-preview {{ width: 100%; }}
  #pc-pano-preview .pc-stage {{
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    overflow: hidden;
    background: var(--muted);
  }}
  #pc-pano-preview .pc-stage img {{
    position: absolute;
    inset: 0;
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }}
  #pc-pano-preview .pc-layer {{ pointer-events: none; }}
</style>
<script>
(() => {{
  const root = document.getElementById('pc-pano-preview');
  const opacity = root.querySelector('#pc-opacity');
  const opacityValue = root.querySelector('#pc-opacity-value');
  const contextToggle = root.querySelector('#pc-context-toggle');
  const targetToggle = root.querySelector('#pc-target-toggle');
  const context = root.querySelector('#pc-context');
  const target = root.querySelector('#pc-target');
  const update = () => {{
    const value = Number(opacity.value) / 100;
    opacityValue.value = `${{opacity.value}}%`;
    context.style.opacity = contextToggle.checked ? String(value) : '0';
    target.style.opacity = targetToggle.checked ? '1' : '0';
  }};
  opacity.addEventListener('input', update);
  contextToggle.addEventListener('change', update);
  targetToggle.addEventListener('change', update);
  update();
}})();
</script>
"""
    html_path.write_text(fragment, encoding="utf-8")


def render(
    tree_id: str,
    view_index: int,
    target_radius_m: float,
    merge_window_frames: int = 0,
    context_radius_m: float | None = None,
    context_color: tuple[int, int, int] | None = None,
    las_output: Path | None = None,
) -> dict[str, object]:
    metadata_path = VIEW_CACHE / tree_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shot = metadata["shots"][view_index - 1]
    frame_id = str(shot["frame_id"])
    frame = load_frame(frame_id)
    color_clouds = color_clouds_in_window(frame_id, merge_window_frames)
    point_arrays = [read_binary_xyzi_rgb(path) for path, _ in color_clouds]
    points = point_arrays[0] if len(point_arrays) == 1 else np.concatenate(point_arrays)
    pcd_path, pcd_image_number = color_clouds[0]

    output_dir = OUTPUT_ROOT / f"{tree_id}_view_{view_index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_path = VIEW_CACHE / tree_id / shot["filename"]
    with Image.open(crop_path) as source:
        background = source.convert("RGB")
    width, height = background.size

    origin_x, origin_y, _ = load_origin()
    camera_x = float(frame["local_x"])
    camera_y = float(frame["local_y"])
    camera_z = float(frame["local_z"])
    camera_heading = float(frame["heading_deg"])
    tree_x = float(metadata["tree"]["easting"]) - origin_x
    tree_y = float(metadata["tree"]["northing"]) - origin_y

    x = points["x"].astype(np.float64)
    y = points["y"].astype(np.float64)
    z = points["z"].astype(np.float64)
    dx = x - camera_x
    dy = y - camera_y
    dz = z - camera_z
    horizontal_distance = np.hypot(dx, dy)
    distance_3d = np.sqrt(dx * dx + dy * dy + dz * dz)
    finite = np.isfinite(distance_3d) & (horizontal_distance > 0.05) & (distance_3d < 70.0)
    target_distance = np.hypot(x - tree_x, y - tree_y)
    if context_radius_m is not None:
        finite &= target_distance <= context_radius_m

    world_bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    relative_bearing = signed_angle_deg(world_bearing - camera_heading)
    panorama_width = float(shot["crop"]["panorama_width"])
    panorama_height = float(shot["crop"]["panorama_height"])
    crop_size = float(shot["crop"]["crop_size_px"])
    center_x = float(shot["crop"]["center_x_px"])
    top = float(shot["crop"]["top_px"])
    output_scale = width / crop_size

    point_panorama_x = (0.5 + relative_bearing / 360.0) * panorama_width
    delta_x = (point_panorama_x - center_x + panorama_width / 2.0) % panorama_width - panorama_width / 2.0
    elevation = np.degrees(np.arctan2(dz, horizontal_distance))
    point_panorama_y = panorama_height * (0.5 - elevation / 180.0)
    u = (delta_x + crop_size / 2.0) * output_scale
    v = (point_panorama_y - top) * output_scale
    visible = finite & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    packed = points["rgb"]
    rgb = np.column_stack(
        (
            (packed >> 16) & 255,
            (packed >> 8) & 255,
            packed & 255,
        )
    ).astype(np.uint8)
    target = visible & (target_distance <= target_radius_m)
    context = visible & ~target

    # Keep the preview readable and deterministic while preserving nearby detail.
    context_indices = np.flatnonzero(context)[:: max(1, int(context.sum() / 22000))]
    target_indices = np.flatnonzero(target)[:: max(1, int(target.sum() / 12000))]
    context_colors = rgb[context_indices]
    if context_color is not None:
        context_colors = np.repeat(
            np.array([context_color], dtype=np.uint8), len(context_indices), axis=0
        )
    point_layer = draw_points(
        width,
        height,
        np.rint(u[context_indices]).astype(int),
        np.rint(v[context_indices]).astype(int),
        context_colors,
        radius=1,
        alpha=150,
    )
    target_colors = np.repeat(np.array([[0, 255, 238]], dtype=np.uint8), len(target_indices), axis=0)
    target_layer = draw_points(
        width,
        height,
        np.rint(u[target_indices]).astype(int),
        np.rint(v[target_indices]).astype(int),
        target_colors,
        radius=2,
        alpha=225,
    )

    las_path: Path | None = None
    if las_output is not None:
        origin_x, origin_y, origin_z = load_origin()
        export_indices = np.concatenate((context_indices, target_indices))
        export_xyz = np.column_stack(
            (
                x[export_indices] + origin_x,
                y[export_indices] + origin_y,
                z[export_indices] + origin_z,
            )
        )
        export_colors = np.concatenate((context_colors, target_colors), axis=0)
        export_classes = np.concatenate(
            (
                np.ones(len(context_indices), dtype=np.uint8),
                np.full(len(target_indices), 5, dtype=np.uint8),
            )
        )
        las_path = las_output.resolve()
        write_las_12_rgb(las_path, export_xyz, export_colors, export_classes)

    # Mark the inventory tree coordinate at an assumed trunk-base elevation. The vertical
    # line is intentionally approximate because the inventory does not contain tree height.
    target_bearing = math.degrees(math.atan2(tree_x - camera_x, tree_y - camera_y)) % 360.0
    target_relative = float(signed_angle_deg(target_bearing - camera_heading))
    target_panorama_x = (0.5 + target_relative / 360.0) * panorama_width
    target_delta_x = (
        (target_panorama_x - center_x + panorama_width / 2.0) % panorama_width
        - panorama_width / 2.0
    )
    target_u = int(round((target_delta_x + crop_size / 2.0) * output_scale))
    target_draw = ImageDraw.Draw(target_layer, "RGBA")
    if 0 <= target_u < width:
        target_draw.line((target_u, 0, target_u, height), fill=(255, 205, 0, 125), width=2)

    background_path = output_dir / "panorama_crop.jpg"
    point_layer_path = output_dir / "pointcloud_rgb.png"
    target_layer_path = output_dir / "target_tree_points.png"
    composite_path = output_dir / "composite.jpg"
    background.save(background_path, quality=88, optimize=True)
    point_layer.save(point_layer_path, optimize=True)
    target_layer.save(target_layer_path, optimize=True)
    composite = Image.alpha_composite(
        Image.alpha_composite(background.convert("RGBA"), point_layer), target_layer
    ).convert("RGB")
    composite.save(composite_path, quality=90, optimize=True)

    report = {
        "tree_id": tree_id,
        "view_index": view_index,
        "frame_id": frame_id,
        "panorama": shot["source_panorama_relpath"],
        "pointcloud": str(pcd_path),
        "pointcloud_files": [str(path) for path, _ in color_clouds],
        "merged_pointcloud_file_count": len(color_clouds),
        "merge_window_frames": merge_window_frames,
        "pointcloud_image_number": pcd_image_number,
        "pointcloud_frame_delta": pcd_image_number - int(frame_id),
        "camera_local_xyz": [camera_x, camera_y, camera_z],
        "camera_heading_deg": camera_heading,
        "tree_local_xy": [tree_x, tree_y],
        "tree_distance_m": float(shot["distance_m"]),
        "target_radius_m": target_radius_m,
        "context_radius_m": context_radius_m,
        "context_color": (
            "#" + "".join(f"{channel:02x}" for channel in context_color)
            if context_color is not None
            else "source_rgb"
        ),
        "source_point_count": int(len(points)),
        "visible_projected_point_count": int(visible.sum()),
        "visible_target_point_count": int(target.sum()),
        "rendered_context_point_count": int(len(context_indices)),
        "rendered_target_point_count": int(len(target_indices)),
        "exported_las_point_count": int(len(context_indices) + len(target_indices)),
        "assumptions": [
            "PCD coordinates are local ENU and share GeoReference_Transformation_Matrix origin.",
            "Ladybug panorama horizontal center is aligned to INS vehicle heading.",
            "Equirectangular panorama vertical center is the local horizon.",
            "No explicit LiDAR-to-Ladybug boresight or lever arm is applied.",
            "INS roll and pitch are not applied in this diagnostic preview.",
        ],
        "outputs": {
            "background": str(background_path),
            "point_layer": str(point_layer_path),
            "target_layer": str(target_layer_path),
            "composite": str(composite_path),
            "las": str(las_path) if las_path is not None else None,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-id", default="RTR0121900")
    parser.add_argument("--view", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--target-radius", type=float, default=5.0)
    parser.add_argument("--merge-window", type=int, default=0)
    parser.add_argument("--context-radius", type=float)
    parser.add_argument("--context-color", type=parse_hex_color)
    parser.add_argument("--las-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args()
    report = render(
        args.tree_id,
        args.view,
        args.target_radius,
        merge_window_frames=args.merge_window,
        context_radius_m=args.context_radius,
        context_color=args.context_color,
        las_output=args.las_output,
    )
    if args.html_output:
        write_inline_preview(report, args.html_output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
