from __future__ import annotations

import sys
import unittest
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    choose_validation_roads,
    select_balanced_records,
    stable_sample_seed,
)


def records(road: str, labels: list[int], *, official_test: bool = False):
    return [
        {
            "sample_key": f"{road}_1_{index}",
            "road_id": road,
            "trajectory_id": "1",
            "tree_id": index,
            "benchmark_label_id": label,
            "is_official_test": official_test,
        }
        for index, label in enumerate(labels, start=1)
    ]


class DatasetPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instances = []
        self.instances += records("06", [1, 1, 2, 2])
        self.instances += records("23", [1, 1, 13, 13])
        self.instances += records("30", [1, 1, 2, 2, 13, 13])
        self.instances += records("08", [1, 2, 13], official_test=True)

    def test_validation_roads_exclude_official_test_roads(self) -> None:
        chosen = choose_validation_roads(
            self.instances,
            (1, 2, 13),
            {1: 1, 2: 1, 13: 1},
            excluded_roads={"08"},
        )
        self.assertNotIn("08", chosen)
        chosen_rows = [item for item in self.instances if item["road_id"] in chosen]
        for label in (1, 2, 13):
            self.assertGreaterEqual(
                sum(item["benchmark_label_id"] == label for item in chosen_rows), 1
            )

    def test_balanced_selection_is_deterministic_and_disjoint(self) -> None:
        quotas = {
            "train": {1: 1, 2: 1, 13: 1},
            "val": {1: 1, 2: 1, 13: 1},
            "test": {1: 1, 2: 1, 13: 1},
        }
        names = {1: "one", 2: "two", 13: "thirteen"}
        first, summary = select_balanced_records(
            self.instances, (1, 2, 13), names, ("06", "23"), quotas, seed=7
        )
        second, _ = select_balanced_records(
            self.instances, (1, 2, 13), names, ("06", "23"), quotas, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(summary["selected_count"], 9)
        self.assertEqual(len({item["sample_key"] for item in first}), 9)
        self.assertEqual({item["split"] for item in first}, {"train", "val", "test"})
        self.assertEqual(stable_sample_seed("sample", 7), stable_sample_seed("sample", 7))


if __name__ == "__main__":
    unittest.main()
