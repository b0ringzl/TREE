"""Coarse WHU-STree trajectory pose matching and equirectangular projection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    image_name: str
    position: tuple[float, float, float]
    roll_deg: float
    pitch_deg: float
    heading_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "image_name": self.image_name,
            "position": list(self.position),
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "heading_deg": self.heading_deg,
        }


@dataclass(frozen=True)
class ProjectionCrop:
    left_unwrapped_px: float
    top_px: float
    width_px: float
    height_px: float
    arc_start_px: float
    arc_width_px: float
    vertical_low_px: float
    vertical_high_px: float

    def to_dict(self) -> dict[str, float]:
        return {
            "left_unwrapped_px": self.left_unwrapped_px,
            "top_px": self.top_px,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "arc_start_px": self.arc_start_px,
            "arc_width_px": self.arc_width_px,
            "vertical_low_px": self.vertical_low_px,
            "vertical_high_px": self.vertical_high_px,
        }


def read_trajectory_csv(path: str | Path) -> tuple[CameraPose, ...]:
    """Read headerless rows ordered as image, X, Y, Z, roll, pitch, heading."""
    csv_path = Path(path)
    poses = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row:
                continue
            if len(row) != 7:
                raise ValueError(f"Expected 7 columns at {csv_path}:{line_number}, got {len(row)}")
            try:
                values = [float(value) for value in row[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric pose at {csv_path}:{line_number}") from exc
            poses.append(
                CameraPose(
                    image_name=row[0],
                    position=tuple(values[:3]),
                    roll_deg=values[3],
                    pitch_deg=values[4],
                    heading_deg=values[5],
                )
            )
    if not poses:
        raise ValueError(f"No poses found in {csv_path}")
    return tuple(poses)


def _wrapped_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def orientation_column_diagnostics(path: str | Path) -> dict[str, object]:
    """Compare each raw orientation column with the local trajectory bearing."""
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if row:
                rows.append([float(value) for value in row[1:]])
    values = np.asarray(rows, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("At least two trajectory records are required")

    xy = values[:, :2]
    delta = np.empty_like(xy)
    delta[0] = xy[1] - xy[0]
    delta[-1] = xy[-1] - xy[-2]
    if len(xy) > 2:
        delta[1:-1] = xy[2:] - xy[:-2]
    path_bearing = np.degrees(np.arctan2(delta[:, 0], delta[:, 1]))

    diagnostics = []
    for column in range(3):
        errors = np.abs(_wrapped_degrees(values[:, 3 + column] - path_bearing))
        diagnostics.append(
            {
                "raw_orientation_column": column + 1,
                "mean_abs_error_deg": float(errors.mean()),
                "median_abs_error_deg": float(np.median(errors)),
                "max_abs_error_deg": float(errors.max()),
            }
        )
    best = min(diagnostics, key=lambda item: item["median_abs_error_deg"])
    return {
        "record_count": len(values),
        "diagnostics": diagnostics,
        "heading_column": best["raw_orientation_column"],
    }


def nearest_poses(
    point: np.ndarray | Iterable[float],
    poses: Iterable[CameraPose],
    count: int = 1,
) -> tuple[tuple[CameraPose, float], ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    target = np.asarray(point, dtype=np.float64)
    if target.shape != (3,):
        raise ValueError(f"Expected a 3D point, got {target.shape}")
    ranked = []
    for pose in poses:
        distance = float(np.linalg.norm(target - np.asarray(pose.position)))
        ranked.append((pose, distance))
    return tuple(sorted(ranked, key=lambda item: item[1])[:count])


def project_equirectangular(
    points_xyz: np.ndarray,
    pose: CameraPose,
    width: int,
    height: int,
    *,
    apply_tilt: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project local East/North/Up points to a full equirectangular panorama."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an N x 3 point array, got {points.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("Panorama dimensions must be positive")

    heading = np.deg2rad(pose.heading_deg)
    pitch = np.deg2rad(pose.pitch_deg if apply_tilt else 0.0)
    roll = np.deg2rad(pose.roll_deg if apply_tilt else 0.0)

    right_level = np.array([np.cos(heading), -np.sin(heading), 0.0])
    forward_level = np.array([np.sin(heading), np.cos(heading), 0.0])
    up_level = np.array([0.0, 0.0, 1.0])

    forward = np.cos(pitch) * forward_level + np.sin(pitch) * up_level
    up_pitched = -np.sin(pitch) * forward_level + np.cos(pitch) * up_level
    right = np.cos(roll) * right_level + np.sin(roll) * up_pitched
    up = -np.sin(roll) * right_level + np.cos(roll) * up_pitched

    relative = points - np.asarray(pose.position, dtype=np.float64)
    camera_right = relative @ right
    camera_forward = relative @ forward
    camera_up = relative @ up
    horizontal = np.hypot(camera_right, camera_forward)
    distance = np.sqrt(horizontal**2 + camera_up**2)

    azimuth = np.arctan2(camera_right, camera_forward)
    elevation = np.arctan2(camera_up, horizontal)
    u = ((azimuth / (2.0 * np.pi)) + 0.5) * width
    v = (0.5 - elevation / np.pi) * height
    return np.mod(u, width), v, distance


def minimal_circular_interval(values: np.ndarray, period: float) -> tuple[float, float]:
    """Return start and width of the shortest circular interval containing values."""
    if period <= 0:
        raise ValueError("period must be positive")
    normalized = np.sort(np.mod(np.asarray(values, dtype=np.float64), period))
    if normalized.size == 0:
        raise ValueError("At least one circular value is required")
    if normalized.size == 1:
        return float(normalized[0]), 0.0
    extended = np.concatenate((normalized, normalized[:1] + period))
    gaps = np.diff(extended)
    largest_gap_index = int(np.argmax(gaps))
    start = float(normalized[(largest_gap_index + 1) % len(normalized)])
    width = float(period - gaps[largest_gap_index])
    return start, width


def projection_crop_bounds(
    u: np.ndarray,
    v: np.ndarray,
    panorama_width: int,
    panorama_height: int,
    *,
    margin: float = 1.25,
    minimum_size: float = 256.0,
    vertical_percentiles: tuple[float, float] = (0.5, 99.5),
) -> ProjectionCrop:
    """Build a seam-aware robust crop around projected tree points."""
    u_values = np.asarray(u, dtype=np.float64)
    v_values = np.asarray(v, dtype=np.float64)
    if u_values.ndim != 1 or v_values.ndim != 1 or u_values.shape != v_values.shape:
        raise ValueError("u and v must be same-length one-dimensional arrays")
    if len(u_values) == 0 or not np.isfinite(u_values).all() or not np.isfinite(v_values).all():
        raise ValueError("Projected coordinates must be non-empty and finite")
    if panorama_width <= 0 or panorama_height <= 0:
        raise ValueError("Panorama dimensions must be positive")
    if margin < 1.0 or minimum_size <= 0:
        raise ValueError("margin must be at least one and minimum_size must be positive")
    low_percentile, high_percentile = vertical_percentiles
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Invalid vertical percentiles")

    arc_start, arc_width = minimal_circular_interval(u_values, panorama_width)
    vertical_low, vertical_high = np.percentile(
        v_values, [low_percentile, high_percentile]
    )
    crop_width = min(
        float(panorama_width), max(minimum_size, arc_width * margin)
    )
    crop_height = min(
        float(panorama_height),
        max(minimum_size, (vertical_high - vertical_low) * margin),
    )
    crop_left = arc_start + arc_width / 2.0 - crop_width / 2.0
    crop_top = (vertical_low + vertical_high) / 2.0 - crop_height / 2.0
    crop_top = max(0.0, min(crop_top, panorama_height - crop_height))
    return ProjectionCrop(
        left_unwrapped_px=float(crop_left),
        top_px=float(crop_top),
        width_px=float(crop_width),
        height_px=float(crop_height),
        arc_start_px=float(arc_start),
        arc_width_px=float(arc_width),
        vertical_low_px=float(vertical_low),
        vertical_high_px=float(vertical_high),
    )
