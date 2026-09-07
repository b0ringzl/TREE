"""Diagnose the raw RIEGL sensor-axis mapping against an existing mapped cloud."""

from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parent
VMMS_ROOT = PROJECT_ROOT.parent.parent / "vmms" / "2023-10-27_hewentian"
SCAN_DIR = VMMS_ROOT / "pointcloud" / "scans"
COLOR_DIR = VMMS_ROOT / "pointcloud" / "ColorCloudPoint"
TRAJECTORY = VMMS_ROOT / "pointcloud" / "mapping" / "INS_trajectory_V.txt"


def read_binary_pcd(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as stream:
        header = b""
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"Incomplete PCD header: {path}")
            header += line
            if line.startswith(b"DATA "):
                break
        match = re.search(rb"^POINTS\s+(\d+)\s*$", header, re.MULTILINE)
        if not match or b"DATA binary" not in header:
            raise RuntimeError(f"Unsupported PCD: {path}")
        count = int(match.group(1))
        payload = stream.read(count * dtype.itemsize)
    if len(payload) != count * dtype.itemsize:
        raise RuntimeError(f"Truncated PCD: {path}")
    return np.frombuffer(payload, dtype=dtype)


RAW_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
        ("ring", "<u2"),
        ("time", "<f4"),
    ]
)
COLOR_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]
)


def xyz(records: np.ndarray) -> np.ndarray:
    return np.column_stack((records["x"], records["y"], records["z"])).astype(
        np.float64
    )


def load_pose(timestamp: float) -> np.ndarray:
    best: tuple[float, np.ndarray] | None = None
    with TRAJECTORY.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line or not line[0].isdigit():
                continue
            values = np.fromstring(line, sep=" ")
            if len(values) < 7:
                continue
            delta = abs(float(values[0]) - timestamp)
            if best is None or delta < best[0]:
                best = (delta, values[1:7])
            if values[0] > timestamp + 0.02:
                break
    if best is None or best[0] > 0.011:
        raise RuntimeError(f"No trajectory pose near {timestamp:.6f}")
    return best[1]


def world_from_flu(roll: float, pitch: float, heading: float) -> np.ndarray:
    """Return ENU-from-forward/left/up using INS radians."""
    sh, ch = math.sin(heading), math.cos(heading)
    level = np.array([[sh, -ch, 0.0], [ch, sh, 0.0], [0.0, 0.0, 1.0]])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    roll_matrix = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    pitch_matrix = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return level @ roll_matrix @ pitch_matrix


def proper_signed_permutations() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for body_axis, sensor_axis in enumerate(permutation):
                matrix[body_axis, sensor_axis] = signs[body_axis]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return matrices


def matrix_description(matrix: np.ndarray) -> str:
    sensor_names = ("sx", "sy", "sz")
    parts = []
    for body_name, row in zip(("forward", "left", "up"), matrix):
        index = int(np.argmax(np.abs(row)))
        sign = "+" if row[index] > 0 else "-"
        parts.append(f"{body_name}={sign}{sensor_names[index]}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=float, default=17840.60)
    parser.add_argument(
        "--mapped-cloud",
        type=Path,
        default=COLOR_DIR / "4319_img4335_LiD102044_ColorizedCloud.pcd",
    )
    args = parser.parse_args()

    scan_path = SCAN_DIR / f"{args.timestamp:.6f}.pcd"
    raw = xyz(read_binary_pcd(scan_path, RAW_DTYPE))
    raw_range = np.linalg.norm(raw, axis=1)
    raw = raw[np.isfinite(raw).all(axis=1) & (raw_range > 1.5) & (raw_range < 70.0)]
    raw = raw[::4]

    mapped = xyz(read_binary_pcd(args.mapped_cloud, COLOR_DTYPE))
    pose = load_pose(args.timestamp)
    position = pose[:3]
    mapped_distance = np.linalg.norm(mapped - position, axis=1)
    mapped = mapped[np.isfinite(mapped).all(axis=1) & (mapped_distance < 75.0)]
    mapped = mapped[::2]
    tree = cKDTree(mapped)
    rotation = world_from_flu(*pose[3:6])

    results = []
    for matrix in proper_signed_permutations():
        transformed = position + (rotation @ matrix @ raw.T).T
        distances, _ = tree.query(transformed, k=1, workers=-1)
        results.append(
            (
                float(np.median(distances)),
                float(np.quantile(distances, 0.75)),
                float(np.mean(np.minimum(distances, 5.0))),
                matrix,
            )
        )
    results.sort(key=lambda item: (item[0], item[1]))
    print(f"raw={len(raw):,} mapped={len(mapped):,} timestamp={args.timestamp:.3f}")
    for rank, (median, q75, clipped_mean, matrix) in enumerate(results[:12], start=1):
        print(
            f"{rank:2d}. median={median:7.3f} q75={q75:7.3f} "
            f"clipped_mean={clipped_mean:7.3f}  {matrix_description(matrix)}"
        )


if __name__ == "__main__":
    main()
