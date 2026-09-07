from __future__ import annotations

import sys
import unittest
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    choose_minimum_trajectory_cover,
    select_shared_smoke_records,
    validate_shared_asset_paths,
)


def record(
    road: str,
    trajectory: str,
    tree: int,
    label: int,
    *,
    official: bool = False,
) -> dict[str, object]:
    sample_key = f"{road}_{trajectory}_{tree}"
    return {
        "sample_key": sample_key,
        "road_id": road,
        "trajectory_id": trajectory,
        "tree_id": tree,
        "benchmark_label_id": label,
        "is_official_test": int(official),
        "point_path": f"assets/points/{sample_key}.npz",
        "image_path": f"assets/images/{sample_key}.jpg",
    }


class SharedAssetPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            record("01", "1", 1, 0),
            record("01", "1", 2, 1),
            record("02", "1", 3, 1),
            record("02", "1", 4, 2),
            record("03", "1", 5, 0),
            record("03", "1", 6, 2),
            record("09", "1", 7, 0, official=True),
            record("09", "1", 8, 1, official=True),
            record("09", "1", 9, 2, official=True),
        ]

    def test_minimum_cover_excludes_official_test(self) -> None:
        cover = choose_minimum_trajectory_cover(
            self.records, (0, 1, 2), max_trajectories=2
        )
        self.assertEqual(cover, (("01", "1"), ("02", "1")))
        self.assertNotIn(("09", "1"), cover)

    def test_smoke_selection_is_deterministic_and_class_complete(self) -> None:
        first, summary = select_shared_smoke_records(
            self.records,
            (0, 1, 2),
            samples_per_class=1,
            max_trajectories=2,
            seed=17,
        )
        second, _ = select_shared_smoke_records(
            self.records,
            (0, 1, 2),
            samples_per_class=1,
            max_trajectories=2,
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual({int(item["benchmark_label_id"]) for item in first}, {0, 1, 2})
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["trajectory_count"], 2)

    def test_noncanonical_shared_path_is_rejected(self) -> None:
        bad = record("01", "1", 1, 0)
        bad["point_path"] = "../points/escape.npz"
        with self.assertRaises(ValueError):
            validate_shared_asset_paths(bad)


if __name__ == "__main__":
    unittest.main()
