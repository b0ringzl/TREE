from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (
    CameraPose,
    minimal_circular_interval,
    nearest_poses,
    orientation_column_diagnostics,
    projection_crop_bounds,
    project_equirectangular,
    read_trajectory_csv,
)


class PanoramaTests(unittest.TestCase):
    def test_cardinal_projection_with_zero_orientation(self) -> None:
        pose = CameraPose("image.jpg", (0.0, 0.0, 0.0), 0.0, 0.0, 0.0)
        points = np.array(
            [[0, 10, 0], [10, 0, 0], [-10, 0, 0], [0, -10, 0]],
            dtype=np.float64,
        )
        u, v, distance = project_equirectangular(points, pose, 800, 400)
        np.testing.assert_allclose(u, [400, 600, 200, 0], atol=1e-8)
        np.testing.assert_allclose(v, 200, atol=1e-8)
        np.testing.assert_allclose(distance, 10, atol=1e-8)

    def test_heading_rotates_world_x_to_forward(self) -> None:
        pose = CameraPose("image.jpg", (0.0, 0.0, 0.0), 0.0, 0.0, 90.0)
        u, _, _ = project_equirectangular(
            np.array([[10.0, 0.0, 0.0]]), pose, 800, 400
        )
        self.assertAlmostEqual(float(u[0]), 400.0)

    def test_minimal_interval_handles_panorama_seam(self) -> None:
        start, width = minimal_circular_interval(np.array([799.0, 1.0, 2.0]), 800.0)
        self.assertAlmostEqual(start, 799.0)
        self.assertAlmostEqual(width, 3.0)

    def test_projection_crop_is_seam_aware_and_vertically_clamped(self) -> None:
        crop = projection_crop_bounds(
            np.array([990.0, 5.0, 10.0]),
            np.array([-20.0, 100.0, 120.0]),
            1000,
            500,
            minimum_size=100.0,
            vertical_percentiles=(0.0, 100.0),
        )
        self.assertAlmostEqual(crop.arc_width_px, 20.0)
        self.assertEqual(crop.width_px, 100.0)
        self.assertEqual(crop.top_px, 0.0)

    def test_pose_reader_and_nearest_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "traj.csv"
            path.write_text(
                "a.jpg,0,0,0,0,1,0\n"
                "b.jpg,0,10,0,0,1,0\n",
                encoding="ascii",
            )
            poses = read_trajectory_csv(path)
            nearest = nearest_poses([0, 8, 0], poses, count=1)
            self.assertEqual(nearest[0][0].image_name, "b.jpg")
            self.assertAlmostEqual(nearest[0][1], 2.0)

    def test_orientation_diagnostics_identifies_third_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "traj.csv"
            path.write_text(
                "a.jpg,0,0,0,1,2,0\n"
                "b.jpg,0,10,0,1,2,0\n"
                "c.jpg,0,20,0,1,2,0\n",
                encoding="ascii",
            )
            diagnostics = orientation_column_diagnostics(path)
            self.assertEqual(diagnostics["heading_column"], 3)


if __name__ == "__main__":
    unittest.main()
