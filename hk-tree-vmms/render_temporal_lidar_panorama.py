"""Project temporally accumulated raw LiDAR scans onto one Ladybug panorama.

The primary window is bounded by the temporal midpoints to the previous and next
panorama.  Every raw scan therefore belongs to exactly one panorama.  Optional
larger symmetric windows are rendered as density/alignment diagnostics.

The equirectangular projection follows the acquisition-side implementation:

    alpha = atan2(y, x)
    omega = asin(z / distance)
    u = 0.5 - alpha / (2*pi)
    v = 0.5 - omega / pi

Camera coordinates are forward/left/up (FLU).  The dataset does not contain an
explicit LiDAR-to-Ladybug boresight/lever-arm file, so the default transform is
the evidence-backed platform-frame approximation and is labelled accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from diagnose_lidar_camera_axes import RAW_DTYPE, read_binary_pcd, world_from_flu


PROJECT_ROOT = Path(__file__).resolve().parent
TREE_ROOT = PROJECT_ROOT.parent
RECEIVE_ROOT = TREE_ROOT.parent
VMMS_ROOT = RECEIVE_ROOT / "vmms" / "2023-10-27_hewentian"
FRAME_CSV = PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv"
SCAN_DIR = VMMS_ROOT / "pointcloud" / "scans"
TRAJECTORY = VMMS_ROOT / "pointcloud" / "mapping" / "INS_trajectory_V.txt"
OUTPUT_ROOT = PROJECT_ROOT / "derived" / "temporal_lidar_projection"
DEFAULT_CAMERA_YAW_OFFSET_DEG = 180.0

# Raw RIEGL points lie on sx/sy scan planes (sz is almost zero).  The mobile
# mapping installation turns that plane upright: vehicle forward=-sz,
# left=+sy, up=+sx.  Besides matching the scanner geometry, this mapping gives
# the best q75 and clipped-mean nearest-neighbour agreement with the supplied
# mapped ColorCloudPoint product.
SENSOR_TO_BODY = np.array(
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


@dataclass(frozen=True)
class Frame:
    frame_id: str
    utc_seconds: float
    image_path: Path
    position: np.ndarray
    attitude: np.ndarray


@dataclass(frozen=True)
class Window:
    label: str
    start: float
    end: float


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["msyhbd.ttc", "msyh.ttc"] if bold else ["msyh.ttc", "segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_frames() -> list[Frame]:
    frames: list[Frame] = []
    with FRAME_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["stream_id"] != "hewentian_pano":
                continue
            image_path = RECEIVE_ROOT / "vmms" / row["source_image_relpath"]
            frames.append(
                Frame(
                    frame_id=row["frame_id"],
                    utc_seconds=float(row["utc_seconds"]),
                    image_path=image_path,
                    position=np.array(
                        [float(row["local_x"]), float(row["local_y"]), float(row["local_z"])],
                        dtype=np.float64,
                    ),
                    attitude=np.array(
                        [float(row["roll_rad"]), float(row["pitch_rad"]), float(row["heading_rad"])],
                        dtype=np.float64,
                    ),
                )
            )
    frames.sort(key=lambda item: item.utc_seconds)
    return frames


def load_trajectory() -> np.ndarray:
    trajectory = np.loadtxt(TRAJECTORY, dtype=np.float64, usecols=range(7))
    trajectory[:, 4:7] = np.unwrap(trajectory[:, 4:7], axis=0)
    return trajectory


def interpolate_pose(trajectory: np.ndarray, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
    times = trajectory[:, 0]
    if timestamp < times[0] or timestamp > times[-1]:
        raise RuntimeError(f"Timestamp {timestamp:.6f} is outside the INS trajectory")
    values = np.array(
        [np.interp(timestamp, times, trajectory[:, column]) for column in range(1, 7)],
        dtype=np.float64,
    )
    return values[:3], values[3:]


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def body_from_camera(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return platform-FLU from camera-FLU for optional empirical offsets."""
    return (
        rotation_z(math.radians(yaw_deg))
        @ rotation_y(math.radians(pitch_deg))
        @ rotation_x(math.radians(roll_deg))
    )


def list_scans() -> list[tuple[float, Path]]:
    result: list[tuple[float, Path]] = []
    pattern = re.compile(r"^(\d+\.\d+)\.pcd$")
    for path in SCAN_DIR.glob("*.pcd"):
        match = pattern.match(path.name)
        if match:
            result.append((float(match.group(1)), path))
    result.sort(key=lambda item: item[0])
    return result


def scans_in_window(scans: list[tuple[float, Path]], window: Window) -> list[tuple[float, Path]]:
    # A file timestamp is its scan start; its point times extend by about 9.3 ms.
    return [(timestamp, path) for timestamp, path in scans if timestamp < window.end and timestamp + 0.01 >= window.start]


def depth_colors(depth: np.ndarray) -> np.ndarray:
    stops = np.array(
        [[49, 46, 129], [37, 99, 235], [34, 211, 238], [250, 204, 21], [239, 68, 68]],
        dtype=np.float64,
    )
    normalized = np.clip((depth - 2.0) / 58.0, 0.0, 1.0)
    scaled = normalized * (len(stops) - 1)
    index = np.minimum(scaled.astype(np.int64), len(stops) - 2)
    fraction = scaled - index
    return np.rint(stops[index] * (1.0 - fraction[:, None]) + stops[index + 1] * fraction[:, None]).astype(np.uint8)


def update_zbuffer(
    depth_buffer: np.ndarray,
    color_buffer: np.ndarray,
    pixel_index: np.ndarray,
    depth: np.ndarray,
) -> None:
    if not len(depth):
        return
    order = np.lexsort((depth, pixel_index))
    ordered_pixels = pixel_index[order]
    first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    chosen = order[first]
    chosen_pixels = pixel_index[chosen]
    chosen_depth = depth[chosen]
    nearer = chosen_depth < depth_buffer[chosen_pixels]
    if not np.any(nearer):
        return
    selected_pixels = chosen_pixels[nearer]
    selected_depth = chosen_depth[nearer]
    depth_buffer[selected_pixels] = selected_depth
    color_buffer[selected_pixels] = depth_colors(selected_depth)


def project_window(
    frame: Frame,
    window: Window,
    trajectory: np.ndarray,
    selected_scans: list[tuple[float, Path]],
    output_width: int,
    output_height: int,
    camera_offset: np.ndarray,
    camera_lever_flu: np.ndarray,
    min_range: float,
    max_range: float,
) -> tuple[np.ndarray, dict[str, object]]:
    ref_world_from_body = world_from_flu(*frame.attitude)
    ref_camera_world_position = frame.position + ref_world_from_body @ camera_lever_flu
    ref_world_from_camera = ref_world_from_body @ camera_offset
    camera_from_world = ref_world_from_camera.T

    depth_buffer = np.full(output_width * output_height, np.inf, dtype=np.float32)
    color_buffer = np.zeros((output_width * output_height, 3), dtype=np.uint8)
    file_input_points = 0
    time_assigned_points = 0
    finite_range_points = 0
    projected_points = 0

    for scan_timestamp, path in selected_scans:
        records = read_binary_pcd(path, RAW_DTYPE)
        sensor_xyz = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64)
        file_input_points += len(sensor_xyz)
        absolute_times = scan_timestamp + records["time"].astype(np.float64)
        time_mask = (absolute_times >= window.start) & (absolute_times < window.end)
        time_assigned_points += int(time_mask.sum())
        ranges = np.linalg.norm(sensor_xyz, axis=1)
        valid = time_mask & np.isfinite(sensor_xyz).all(axis=1) & (ranges >= min_range) & (ranges <= max_range)
        sensor_xyz = sensor_xyz[valid]
        finite_range_points += len(sensor_xyz)
        if not len(sensor_xyz):
            continue

        # Use the scan midpoint for motion compensation. Individual point offsets
        # are < 9.3 ms and are retained in the documented temporal span.
        point_offsets = records["time"][valid]
        pose_timestamp = scan_timestamp + float(np.median(point_offsets))
        scan_position, scan_attitude = interpolate_pose(trajectory, pose_timestamp)
        world_from_sensor = world_from_flu(*scan_attitude) @ SENSOR_TO_BODY
        world_xyz = scan_position + (world_from_sensor @ sensor_xyz.T).T
        camera_xyz = (camera_from_world @ (world_xyz - ref_camera_world_position).T).T

        distance = np.linalg.norm(camera_xyz, axis=1)
        valid_camera = np.isfinite(camera_xyz).all(axis=1) & (distance > 1e-6)
        camera_xyz = camera_xyz[valid_camera]
        distance = distance[valid_camera]
        if not len(camera_xyz):
            continue

        alpha = np.arctan2(camera_xyz[:, 1], camera_xyz[:, 0])
        omega = np.arcsin(np.clip(camera_xyz[:, 2] / distance, -1.0, 1.0))
        u = 0.5 - alpha / (2.0 * math.pi)
        v = 0.5 - omega / math.pi
        pixel_u = np.floor(u * output_width + 0.5).astype(np.int64) % output_width
        pixel_v = np.floor(v * output_height + 0.5).astype(np.int64)
        inside = (pixel_v >= 0) & (pixel_v < output_height)
        pixel_u, pixel_v, distance = pixel_u[inside], pixel_v[inside], distance[inside]
        projected_points += len(distance)
        update_zbuffer(depth_buffer, color_buffer, pixel_v * output_width + pixel_u, distance)

    occupied = np.isfinite(depth_buffer)
    rgba = np.zeros((output_height * output_width, 4), dtype=np.uint8)
    rgba[occupied, :3] = color_buffer[occupied]
    rgba[occupied, 3] = 220
    overlay = rgba.reshape(output_height, output_width, 4)
    metrics: dict[str, object] = {
        "scan_count": len(selected_scans),
        "scan_start": selected_scans[0][0] if selected_scans else None,
        "scan_end": selected_scans[-1][0] + 0.01 if selected_scans else None,
        "file_input_points": file_input_points,
        "input_points": time_assigned_points,
        "range_filtered_points": finite_range_points,
        "projected_points": projected_points,
        "occupied_pixels": int(occupied.sum()),
        "occupied_fraction": float(occupied.mean()),
    }
    return overlay, metrics


def make_composite(background: Image.Image, overlay_array: np.ndarray, opacity: float = 0.78) -> tuple[Image.Image, Image.Image]:
    overlay = Image.fromarray(overlay_array, mode="RGBA")
    if opacity < 1.0:
        alpha = overlay.getchannel("A").point(lambda value: int(value * opacity))
        overlay.putalpha(alpha)
    composite = Image.alpha_composite(background.convert("RGBA"), overlay)
    return overlay, composite.convert("RGB")


def make_dashboard(
    items: list[tuple[Window, Image.Image, dict[str, object]]],
    frame: Frame,
    output_width: int,
    yaw_offset_deg: float,
) -> Image.Image:
    header_height = 126
    caption_height = 70
    panel_height = items[0][1].height + caption_height
    canvas = Image.new("RGB", (output_width, header_height + panel_height * len(items)), "#0b1220")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(27, bold=True)
    body_font = load_font(18)
    caption_font = load_font(17, bold=True)
    draw.text((24, 18), "原始 LiDAR 时序累积 → Ladybug 全景影像", font=title_font, fill="#f8fafc")
    draw.text(
        (24, 63),
        f"影像帧 {frame.frame_id}  UTC {frame.utc_seconds:.3f}s｜逐扫描 INS 运动补偿｜全景零方位修正 {yaw_offset_deg:g}°｜采用给定球面公式",
        font=body_font,
        fill="#cbd5e1",
    )
    draw.text(
        (24, 94),
        "颜色：紫/蓝=近，青/黄/红=远；当前使用平台共轴近似，后续可由外参进一步精配准",
        font=body_font,
        fill="#94a3b8",
    )
    y = header_height
    for window, image, metrics in items:
        caption = (
            f"{window.label}｜{metrics['scan_count']:,} 扫描｜{metrics['input_points']:,} 点｜"
            f"{metrics['occupied_pixels']:,} 投影像素｜时间跨度 {window.end - window.start:.3f}s"
        )
        draw.rectangle((0, y, output_width, y + caption_height), fill="#111827")
        draw.text((24, y + 18), caption, font=caption_font, fill="#f1f5f9")
        canvas.paste(image, (0, y + caption_height))
        y += panel_height
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", default="004338")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--half-windows", type=float, nargs="*", default=[0.2, 1.0, 3.0])
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=DEFAULT_CAMERA_YAW_OFFSET_DEG,
        help="Ladybug camera-forward yaw in the INS platform FLU frame (default: 180)",
    )
    parser.add_argument("--pitch-offset-deg", type=float, default=0.0)
    parser.add_argument("--roll-offset-deg", type=float, default=0.0)
    parser.add_argument("--lever-forward-m", type=float, default=0.0)
    parser.add_argument("--lever-left-m", type=float, default=0.0)
    parser.add_argument("--lever-up-m", type=float, default=0.0)
    parser.add_argument("--min-range", type=float, default=1.5)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    frames = load_frames()
    matching = [index for index, frame in enumerate(frames) if frame.frame_id == args.frame_id]
    if not matching:
        raise RuntimeError(f"Frame {args.frame_id} not found")
    frame_index = matching[0]
    if frame_index == 0 or frame_index == len(frames) - 1:
        raise RuntimeError("A midpoint window requires both a previous and next panorama")
    frame = frames[frame_index]
    previous_frame, next_frame = frames[frame_index - 1], frames[frame_index + 1]
    midpoint_window = Window(
        "主窗口：相邻全景帧时间中点分配",
        (previous_frame.utc_seconds + frame.utc_seconds) / 2.0,
        (frame.utc_seconds + next_frame.utc_seconds) / 2.0,
    )
    windows = [midpoint_window]
    for half_window in args.half_windows:
        if half_window <= 0:
            continue
        windows.append(Window(f"对照窗口：±{half_window:g} s", frame.utc_seconds - half_window, frame.utc_seconds + half_window))

    if not frame.image_path.exists():
        raise FileNotFoundError(frame.image_path)
    with Image.open(frame.image_path) as source:
        source.load()
        output_height = round(args.width * source.height / source.width)
        background = source.resize((args.width, output_height), Image.Resampling.LANCZOS).convert("RGB")

    trajectory = load_trajectory()
    all_scans = list_scans()
    camera_offset = body_from_camera(args.yaw_offset_deg, args.pitch_offset_deg, args.roll_offset_deg)
    camera_lever = np.array([args.lever_forward_m, args.lever_left_m, args.lever_up_m], dtype=np.float64)
    output_dir = args.output_root / f"frame_{frame.frame_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    background_path = output_dir / "panorama_1600x800.jpg"
    background.save(background_path, quality=92)

    dashboard_items: list[tuple[Window, Image.Image, dict[str, object]]] = []
    report_windows: list[dict[str, object]] = []
    for index, window in enumerate(windows):
        selected = scans_in_window(all_scans, window)
        overlay_array, metrics = project_window(
            frame,
            window,
            trajectory,
            selected,
            args.width,
            output_height,
            camera_offset,
            camera_lever,
            args.min_range,
            args.max_range,
        )
        overlay, composite = make_composite(background, overlay_array)
        stem = "midpoint" if index == 0 else f"plus_minus_{args.half_windows[index - 1]:g}s".replace(".", "p")
        overlay_path = output_dir / f"{stem}_overlay.png"
        composite_path = output_dir / f"{stem}_composite.jpg"
        overlay.save(overlay_path)
        composite.save(composite_path, quality=92)
        dashboard_items.append((window, composite, metrics))
        report_windows.append(
            {
                "label": window.label,
                "start": window.start,
                "end": window.end,
                **metrics,
                "overlay": str(overlay_path),
                "composite": str(composite_path),
            }
        )
        print(
            f"{window.label}: {metrics['scan_count']} scans, "
            f"{metrics['input_points']:,} points, {metrics['occupied_pixels']:,} occupied pixels"
        )

    dashboard = make_dashboard(dashboard_items, frame, args.width, args.yaw_offset_deg)
    dashboard_path = output_dir / "temporal_accumulation_comparison.jpg"
    dashboard.save(dashboard_path, quality=94)
    report = {
        "frame_id": frame.frame_id,
        "panorama_timestamp": frame.utc_seconds,
        "panorama_path": str(frame.image_path),
        "projection_formula": {
            "alpha": "atan2(y, x)",
            "omega": "asin(z / sqrt(x*x+y*y+z*z))",
            "u": "0.5 - alpha/(2*pi)",
            "v": "0.5 - omega/pi",
        },
        "axis_convention": {
            "camera": "x=forward, y=left, z=up (FLU)",
            "raw_lidar_to_body": "forward=-sensor_z, left=+sensor_y, up=+sensor_x",
        },
        "motion_compensation": "each raw scan transformed by interpolated INS pose into reference panorama pose",
        "extrinsic_status": (
            "verified 180-degree Ladybug panorama zero-azimuth correction applied; "
            "fine LiDAR-Ladybug boresight/lever-arm parameters are not present in supplied files"
        ),
        "camera_offsets_deg": {
            "yaw": args.yaw_offset_deg,
            "pitch": args.pitch_offset_deg,
            "roll": args.roll_offset_deg,
        },
        "camera_lever_flu_m": camera_lever.tolist(),
        "windows": report_windows,
        "dashboard": str(dashboard_path),
    }
    report_path = output_dir / "projection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dashboard: {dashboard_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
