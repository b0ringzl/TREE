from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Iterator, Sequence

import pyproj
from pyproj import Transformer
from pyproj.transformer import TransformerGroup


HK_TZ = timezone(timedelta(hours=8))
UTC = timezone.utc
PANORAMA_FRAME_RE = re.compile(r"_Panoramic_(\d{6})_")


@dataclass(frozen=True)
class StreamConfig:
    stream_id: str
    image_dir: str
    gps_file: str


@dataclass(frozen=True)
class SurveyConfig:
    survey_id: str
    capture_date: date
    folder: str
    trajectory_file: str
    streams: tuple[StreamConfig, ...]
    origin_kind: str
    origin_file: str


@dataclass(frozen=True)
class TrajectoryRow:
    utc_seconds: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    heading: float
    status: float
    sd_x: float
    sd_y: float
    sd_z: float


SURVEYS = (
    SurveyConfig(
        survey_id="2023-10-27_hewentian",
        capture_date=date(2023, 10, 27),
        folder="2023-10-27_hewentian",
        trajectory_file="pointcloud/mapping/INS_trajectory.txt",
        streams=(
            StreamConfig(
                stream_id="hewentian_pano",
                image_dir="ladybug/img_output",
                gps_file="ladybug/img_output/ladybug_frame_gps_info_6088.txt",
            ),
        ),
        origin_kind="matrix",
        origin_file="pointcloud/GeoReference_Transformation_Matrix.txt",
    ),
    SurveyConfig(
        survey_id="2025-02-10_jianshazui",
        capture_date=date(2025, 2, 10),
        folder="20250210-jianshazui",
        trajectory_file="pointcloud/INS_trajectory.txt",
        streams=(
            StreamConfig(
                stream_id="jianshazui_pano_1",
                image_dir="ladybug/Pano",
                gps_file="ladybug/Pano/ladybug_frame_gps_info_3299.txt",
            ),
            StreamConfig(
                stream_id="jianshazui_pano_2",
                image_dir="ladybug/Pano2",
                gps_file="ladybug/Pano2/ladybug_frame_gps_info_7421.txt",
            ),
        ),
        origin_kind="origin_hk80",
        origin_file="pointcloud/Origin_HK80.txt",
    ),
    SurveyConfig(
        survey_id="2024-07-25_stubbs_road",
        capture_date=date(2024, 7, 25),
        folder="Stubbs Road_2024-07-25_HyD_collect",
        trajectory_file="pointcloud/INS_trajectory.txt",
        streams=(
            StreamConfig(
                stream_id="stubbs_pano_0",
                image_dir="ladybug/pano0",
                gps_file=(
                    "ladybug/pano0/"
                    "ladybug_21505973_20240725_175246_gps_info_748.txt"
                ),
            ),
            StreamConfig(
                stream_id="stubbs_pano_1",
                image_dir="ladybug/pano",
                gps_file=(
                    "ladybug/pano/"
                    "ladybug_21505973_20240725_175438_gps_info_3135.txt"
                ),
            ),
        ),
        origin_kind="origin_hk80",
        origin_file="pointcloud/Origin_HK80.txt",
    ),
)


FRAME_COLUMNS = (
    "survey_id",
    "stream_id",
    "capture_date",
    "frame_id",
    "seq_id",
    "utc_seconds",
    "utc_datetime",
    "hong_kong_datetime",
    "source_image_relpath",
    "local_x",
    "local_y",
    "local_z",
    "hk80_easting",
    "hk80_northing",
    "hk80_z",
    "wgs84_latitude",
    "wgs84_longitude",
    "roll_rad",
    "pitch_rad",
    "heading_rad",
    "roll_deg",
    "pitch_deg",
    "heading_deg",
    "trajectory_status",
    "position_sd_x_m",
    "position_sd_y_m",
    "position_sd_z_m",
    "horizontal_position_sd_m",
    "coordinate_role",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve VMMS panorama frames and mapped point clouds to HK80/WGS84."
    )
    parser.add_argument("--vmms-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compute-downsample-bounds",
        action="store_true",
        help="Scan pointcloud_DS_*.pcd files to calculate coordinate bounds.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_origin(survey_root: Path, config: SurveyConfig) -> tuple[float, float, float]:
    path = require_file(survey_root / config.origin_file)
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if config.origin_kind == "matrix":
        if len(values) < 3:
            raise ValueError(f"Invalid transformation matrix: {path}")
        rows = [[float(value) for value in line.split()] for line in values[:3]]
        return rows[0][3], rows[1][3], rows[2][3]
    if config.origin_kind == "origin_hk80":
        if len(values) < 4:
            raise ValueError(f"Invalid HK80 origin: {path}")
        return float(values[1]), float(values[2]), float(values[3])
    raise ValueError(f"Unsupported origin kind: {config.origin_kind}")


def load_trajectory(path: Path) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []
    with require_file(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 11:
                raise ValueError(f"Trajectory row {line_number} has {len(fields)} columns: {path}")
            numbers = [float(value) for value in fields[:11]]
            rows.append(TrajectoryRow(*numbers))
    if len(rows) < 2:
        raise ValueError(f"Trajectory has fewer than two rows: {path}")
    for previous, current in zip(rows, rows[1:]):
        if current.utc_seconds <= previous.utc_seconds:
            raise ValueError(f"Trajectory time is not strictly increasing: {path}")
    return rows


def time_of_day_seconds(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def timestamp_iso(capture_date: date, seconds: float, tz: timezone) -> str:
    base = datetime.combine(capture_date, datetime.min.time(), tzinfo=UTC)
    value = base + timedelta(seconds=seconds)
    if tz is not UTC:
        value = value.astimezone(tz)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def image_frame_map(image_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in image_dir.glob("*.jpg"):
        match = PANORAMA_FRAME_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def read_camera_rows(gps_path: Path, image_dir: Path, vmms_root: Path) -> list[dict[str, object]]:
    images = image_frame_map(image_dir)
    rows: list[dict[str, object]] = []
    with require_file(gps_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for raw_row in reader:
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            frame_id = int(row["FRAME"])
            image_path = images.get(frame_id)
            if image_path is None:
                raise FileNotFoundError(
                    f"No panorama image for frame {frame_id} listed in {gps_path}"
                )
            rows.append(
                {
                    "frame_id": frame_id,
                    "seq_id": int(row["SEQID"]),
                    "utc_seconds": time_of_day_seconds(row["CAMERA TIME"]),
                    "source_image_relpath": image_path.relative_to(vmms_root).as_posix(),
                }
            )
    return rows


def lerp(left: float, right: float, weight: float) -> float:
    return left + (right - left) * weight


def lerp_angle(left: float, right: float, weight: float) -> float:
    delta = (right - left + math.pi) % (2 * math.pi) - math.pi
    return left + delta * weight


def interpolate_trajectory(
    trajectory: Sequence[TrajectoryRow],
    trajectory_times: Sequence[float],
    utc_seconds: float,
) -> TrajectoryRow:
    index = bisect.bisect_left(trajectory_times, utc_seconds)
    if index == 0:
        if utc_seconds < trajectory_times[0]:
            raise ValueError(
                f"Camera time {utc_seconds} is before trajectory start "
                f"{trajectory_times[0]}"
            )
        return trajectory[0]
    if index == len(trajectory):
        raise ValueError(
            f"Camera time {utc_seconds} is after trajectory end {trajectory_times[-1]}"
        )
    if trajectory_times[index] == utc_seconds:
        return trajectory[index]
    left = trajectory[index - 1]
    right = trajectory[index]
    weight = (utc_seconds - left.utc_seconds) / (right.utc_seconds - left.utc_seconds)
    return TrajectoryRow(
        utc_seconds=utc_seconds,
        x=lerp(left.x, right.x, weight),
        y=lerp(left.y, right.y, weight),
        z=lerp(left.z, right.z, weight),
        roll=lerp_angle(left.roll, right.roll, weight),
        pitch=lerp_angle(left.pitch, right.pitch, weight),
        heading=lerp_angle(left.heading, right.heading, weight),
        status=left.status if weight < 0.5 else right.status,
        sd_x=lerp(left.sd_x, right.sd_x, weight),
        sd_y=lerp(left.sd_y, right.sd_y, weight),
        sd_z=lerp(left.sd_z, right.sd_z, weight),
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return lerp(ordered[lower], ordered[upper], position - lower)


def route_distance(rows: Sequence[dict[str, object]]) -> float:
    distance = 0.0
    for left, right in zip(rows, rows[1:]):
        step = math.hypot(
            float(right["hk80_easting"]) - float(left["hk80_easting"]),
            float(right["hk80_northing"]) - float(left["hk80_northing"]),
        )
        if step < 5.0:
            distance += step
    return distance


def solve_stream(
    config: SurveyConfig,
    stream: StreamConfig,
    survey_root: Path,
    vmms_root: Path,
    trajectory: Sequence[TrajectoryRow],
    origin: tuple[float, float, float],
    transformer: Transformer,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    camera_rows = read_camera_rows(
        survey_root / stream.gps_file,
        survey_root / stream.image_dir,
        vmms_root,
    )
    trajectory_times = [row.utc_seconds for row in trajectory]
    solved: list[dict[str, object]] = []
    horizontal_sd: list[float] = []
    vertical_sd: list[float] = []
    for camera in camera_rows:
        utc_seconds = float(camera["utc_seconds"])
        pose = interpolate_trajectory(trajectory, trajectory_times, utc_seconds)
        easting = origin[0] + pose.x
        northing = origin[1] + pose.y
        height = origin[2] + pose.z
        longitude, latitude = transformer.transform(easting, northing)
        h_sd = math.hypot(pose.sd_x, pose.sd_y)
        horizontal_sd.append(h_sd)
        vertical_sd.append(pose.sd_z)
        solved.append(
            {
                "survey_id": config.survey_id,
                "stream_id": stream.stream_id,
                "capture_date": config.capture_date.isoformat(),
                "frame_id": f"{int(camera['frame_id']):06d}",
                "seq_id": int(camera["seq_id"]),
                "utc_seconds": round(utc_seconds, 4),
                "utc_datetime": timestamp_iso(config.capture_date, utc_seconds, UTC),
                "hong_kong_datetime": timestamp_iso(config.capture_date, utc_seconds, HK_TZ),
                "source_image_relpath": camera["source_image_relpath"],
                "local_x": round(pose.x, 6),
                "local_y": round(pose.y, 6),
                "local_z": round(pose.z, 6),
                "hk80_easting": round(easting, 6),
                "hk80_northing": round(northing, 6),
                "hk80_z": round(height, 6),
                "wgs84_latitude": round(latitude, 10),
                "wgs84_longitude": round(longitude, 10),
                "roll_rad": round(pose.roll, 9),
                "pitch_rad": round(pose.pitch, 9),
                "heading_rad": round(pose.heading, 9),
                "roll_deg": round(math.degrees(pose.roll), 6),
                "pitch_deg": round(math.degrees(pose.pitch), 6),
                "heading_deg": round(math.degrees(pose.heading), 6),
                "trajectory_status": pose.status,
                "position_sd_x_m": round(pose.sd_x, 6),
                "position_sd_y_m": round(pose.sd_y, 6),
                "position_sd_z_m": round(pose.sd_z, 6),
                "horizontal_position_sd_m": round(h_sd, 6),
                "coordinate_role": "interpolated_INS_platform_position",
            }
        )
    eastings = [float(row["hk80_easting"]) for row in solved]
    northings = [float(row["hk80_northing"]) for row in solved]
    heights = [float(row["hk80_z"]) for row in solved]
    latitudes = [float(row["wgs84_latitude"]) for row in solved]
    longitudes = [float(row["wgs84_longitude"]) for row in solved]
    summary = {
        "stream_id": stream.stream_id,
        "frame_count": len(solved),
        "utc_seconds_start": solved[0]["utc_seconds"],
        "utc_seconds_end": solved[-1]["utc_seconds"],
        "route_distance_km": round(route_distance(solved) / 1000.0, 4),
        "hk80_bounds": {
            "easting_min": min(eastings),
            "easting_max": max(eastings),
            "northing_min": min(northings),
            "northing_max": max(northings),
            "z_min": min(heights),
            "z_max": max(heights),
        },
        "wgs84_bounds": {
            "latitude_min": min(latitudes),
            "latitude_max": max(latitudes),
            "longitude_min": min(longitudes),
            "longitude_max": max(longitudes),
        },
        "estimated_position_sd_m": {
            "horizontal_median": round(median(horizontal_sd), 4),
            "horizontal_p95": round(percentile(horizontal_sd, 0.95), 4),
            "horizontal_max": round(max(horizontal_sd), 4),
            "vertical_median": round(median(vertical_sd), 4),
            "vertical_p95": round(percentile(vertical_sd, 0.95), 4),
            "vertical_max": round(max(vertical_sd), 4),
        },
        "first_frame": solved[0],
        "middle_frame": solved[len(solved) // 2],
        "last_frame": solved[-1],
    }
    return solved, summary


def read_pcd_header(path: Path) -> dict[str, object]:
    header: dict[str, object] = {}
    with path.open("rb") as handle:
        for _ in range(30):
            raw_line = handle.readline()
            if not raw_line:
                break
            line = raw_line.decode("ascii", errors="replace").strip()
            fields = line.split()
            if len(fields) >= 2:
                header[fields[0].upper()] = fields[1:]
            if fields and fields[0].upper() == "DATA":
                header["header_bytes"] = handle.tell()
                break
    return header


def pcd_binary_expected_bytes(header: dict[str, object]) -> int | None:
    if not header.get("POINTS") or not header.get("SIZE") or not header.get("COUNT"):
        return None
    points = int(header["POINTS"][0])
    sizes = [int(value) for value in header["SIZE"]]
    counts = [int(value) for value in header["COUNT"]]
    return int(header.get("header_bytes", 0)) + points * sum(
        size * count for size, count in zip(sizes, counts)
    )


def ascii_pcd_bounds(path: Path) -> tuple[float, float, float, float, float, float]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    in_data = False
    with path.open("r", encoding="ascii", errors="strict") as handle:
        for line in handle:
            if not in_data:
                if line.upper().startswith("DATA "):
                    if "ASCII" not in line.upper():
                        raise ValueError(f"Point cloud is not ASCII: {path}")
                    in_data = True
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            values = [float(fields[index]) for index in range(3)]
            for index, value in enumerate(values):
                minimum[index] = min(minimum[index], value)
                maximum[index] = max(maximum[index], value)
    if math.isinf(minimum[0]):
        raise ValueError(f"No point rows found: {path}")
    return minimum[0], maximum[0], minimum[1], maximum[1], minimum[2], maximum[2]


def pointcloud_inventory(
    vmms_root: Path,
    configs: Sequence[SurveyConfig],
    origins: dict[str, tuple[float, float, float]],
    transformer: Transformer,
    compute_downsample_bounds: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    assets: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    for config in configs:
        survey_root = vmms_root / config.folder
        mapping_dir = survey_root / "pointcloud" / "mapping"
        if not mapping_dir.is_dir():
            continue
        origin = origins[config.survey_id]
        for path in sorted(mapping_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in {".pcd", ".las"}:
                continue
            row: dict[str, object] = {
                "survey_id": config.survey_id,
                "source_relpath": path.relative_to(vmms_root).as_posix(),
                "format": path.suffix.lower().lstrip("."),
                "actual_bytes": path.stat().st_size,
                "origin_hk80_easting": origin[0],
                "origin_hk80_northing": origin[1],
                "origin_hk80_z": origin[2],
                "coordinate_formula": "absolute_HK80=origin+local_xyz",
                "fields": "",
                "data_encoding": "",
                "point_count": "",
                "expected_binary_bytes": "",
                "integrity_status": "not_checked",
                "local_x_min": "",
                "local_x_max": "",
                "local_y_min": "",
                "local_y_max": "",
                "local_z_min": "",
                "local_z_max": "",
                "hk80_easting_min": "",
                "hk80_easting_max": "",
                "hk80_northing_min": "",
                "hk80_northing_max": "",
                "hk80_z_min": "",
                "hk80_z_max": "",
                "wgs84_latitude_min": "",
                "wgs84_latitude_max": "",
                "wgs84_longitude_min": "",
                "wgs84_longitude_max": "",
            }
            if path.suffix.lower() == ".pcd":
                header = read_pcd_header(path)
                data_encoding = " ".join(header.get("DATA", []))
                row["fields"] = " ".join(header.get("FIELDS", []))
                row["data_encoding"] = data_encoding
                if header.get("POINTS"):
                    row["point_count"] = int(header["POINTS"][0])
                if data_encoding.lower() == "binary":
                    expected = pcd_binary_expected_bytes(header)
                    row["expected_binary_bytes"] = expected or ""
                    if expected:
                        ratio = path.stat().st_size / expected
                        row["integrity_status"] = (
                            "ok" if 0.999 <= ratio <= 1.001 else "size_mismatch"
                        )
                        if row["integrity_status"] != "ok":
                            anomalies.append(
                                {
                                    "survey_id": config.survey_id,
                                    "source_relpath": row["source_relpath"],
                                    "actual_bytes": path.stat().st_size,
                                    "expected_binary_bytes": expected,
                                    "size_ratio": round(ratio, 6),
                                }
                            )
                elif data_encoding.lower() == "ascii":
                    row["integrity_status"] = "header_ok"
                if (
                    compute_downsample_bounds
                    and data_encoding.lower() == "ascii"
                    and "_ds_" in path.name.lower()
                ):
                    bounds = ascii_pcd_bounds(path)
                    row.update(
                        {
                            "local_x_min": bounds[0],
                            "local_x_max": bounds[1],
                            "local_y_min": bounds[2],
                            "local_y_max": bounds[3],
                            "local_z_min": bounds[4],
                            "local_z_max": bounds[5],
                            "hk80_easting_min": origin[0] + bounds[0],
                            "hk80_easting_max": origin[0] + bounds[1],
                            "hk80_northing_min": origin[1] + bounds[2],
                            "hk80_northing_max": origin[1] + bounds[3],
                            "hk80_z_min": origin[2] + bounds[4],
                            "hk80_z_max": origin[2] + bounds[5],
                        }
                    )
                    corners = [
                        transformer.transform(
                            float(row[easting_key]), float(row[northing_key])
                        )
                        for easting_key in ("hk80_easting_min", "hk80_easting_max")
                        for northing_key in ("hk80_northing_min", "hk80_northing_max")
                    ]
                    row["wgs84_longitude_min"] = min(item[0] for item in corners)
                    row["wgs84_longitude_max"] = max(item[0] for item in corners)
                    row["wgs84_latitude_min"] = min(item[1] for item in corners)
                    row["wgs84_latitude_max"] = max(item[1] for item in corners)
                    row["integrity_status"] = "bounds_scanned"
            assets.append(row)
    return assets, anomalies


def geojson_features(stream_rows: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    features: list[dict[str, object]] = []
    for stream_id, rows in stream_rows.items():
        stride = max(1, len(rows) // 500)
        selected = rows[::stride]
        if selected[-1] is not rows[-1]:
            selected.append(rows[-1])
        coordinates = [
            [
                float(row["wgs84_longitude"]),
                float(row["wgs84_latitude"]),
                float(row["hk80_z"]),
            ]
            for row in selected
        ]
        properties = {
            "survey_id": rows[0]["survey_id"],
            "stream_id": stream_id,
            "frame_count": len(rows),
            "start_utc": rows[0]["utc_datetime"],
            "end_utc": rows[-1]["utc_datetime"],
            "coordinate_role": rows[0]["coordinate_role"],
        }
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
        for endpoint, row in (("start", rows[0]), ("end", rows[-1])):
            features.append(
                {
                    "type": "Feature",
                    "properties": {**properties, "endpoint": endpoint},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            float(row["wgs84_longitude"]),
                            float(row["wgs84_latitude"]),
                            float(row["hk80_z"]),
                        ],
                    },
                }
            )
    return features


def write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    vmms_root = args.vmms_root.resolve()
    output_dir = args.output_dir.resolve()
    if not vmms_root.is_dir():
        raise NotADirectoryError(vmms_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    group = TransformerGroup(
        pyproj.CRS.from_epsg(2326),
        pyproj.CRS.from_epsg(4326),
        always_xy=True,
    )
    if not group.best_available or not group.transformers:
        raise RuntimeError("A usable EPSG:2326 to EPSG:4326 transformation is unavailable")
    transformer = group.transformers[0]

    all_rows: list[dict[str, object]] = []
    stream_rows: dict[str, list[dict[str, object]]] = {}
    survey_summaries: list[dict[str, object]] = []
    origins: dict[str, tuple[float, float, float]] = {}

    for config in SURVEYS:
        survey_root = vmms_root / config.folder
        if not survey_root.is_dir():
            raise NotADirectoryError(survey_root)
        origin = parse_origin(survey_root, config)
        origins[config.survey_id] = origin
        trajectory = load_trajectory(survey_root / config.trajectory_file)
        origin_lon, origin_lat = transformer.transform(origin[0], origin[1])
        solved_streams: list[dict[str, object]] = []
        for stream in config.streams:
            solved, stream_summary = solve_stream(
                config,
                stream,
                survey_root,
                vmms_root,
                trajectory,
                origin,
                transformer,
            )
            all_rows.extend(solved)
            stream_rows[stream.stream_id] = solved
            solved_streams.append(stream_summary)
        survey_summaries.append(
            {
                "survey_id": config.survey_id,
                "source_folder": config.folder,
                "capture_date": config.capture_date.isoformat(),
                "trajectory_relpath": config.trajectory_file,
                "trajectory_start_utc_seconds": trajectory[0].utc_seconds,
                "trajectory_end_utc_seconds": trajectory[-1].utc_seconds,
                "trajectory_row_count": len(trajectory),
                "origin_hk80": {
                    "easting": origin[0],
                    "northing": origin[1],
                    "z": origin[2],
                },
                "origin_wgs84": {
                    "latitude": origin_lat,
                    "longitude": origin_lon,
                },
                "streams": solved_streams,
            }
        )

    asset_rows, anomalies = pointcloud_inventory(
        vmms_root,
        SURVEYS,
        origins,
        transformer,
        args.compute_downsample_bounds,
    )

    write_csv(output_dir / "frame_coordinates.csv", all_rows, FRAME_COLUMNS)
    if asset_rows:
        write_csv(output_dir / "pointcloud_assets.csv", asset_rows, tuple(asset_rows[0].keys()))

    geojson = {
        "type": "FeatureCollection",
        "name": "VMMS panorama acquisition routes",
        "crs_note": "RFC 7946 WGS84 longitude/latitude; third coordinate is source HK80 z.",
        "features": geojson_features(stream_rows),
    }
    (output_dir / "route_trajectories.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "vmms_root": str(vmms_root),
        "output_directory": str(output_dir),
        "source_crs": "EPSG:2326 Hong Kong 1980 Grid System",
        "target_crs": "EPSG:4326 WGS 84",
        "coordinate_formula": {
            "hk80_easting": "origin_easting + local_x",
            "hk80_northing": "origin_northing + local_y",
            "hk80_z": "origin_z + local_z",
        },
        "transformation": {
            "description": transformer.description,
            "reported_accuracy_m": transformer.accuracy,
            "best_available": group.best_available,
            "pyproj_version": pyproj.__version__,
            "proj_version": pyproj.proj_version_str,
        },
        "coordinate_role": "interpolated INS/platform position; camera lever arm not applied",
        "frame_count": len(all_rows),
        "pointcloud_asset_count": len(asset_rows),
        "pointcloud_anomalies": anomalies,
        "surveys": survey_summaries,
    }
    (output_dir / "coordinate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Solved panorama frames: {len(all_rows)}")
    print(f"Mapped point-cloud assets: {len(asset_rows)}")
    print(f"Point-cloud anomalies: {len(anomalies)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
