from __future__ import annotations

import sys
import unittest
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    build_road_class_counts,
    choose_inner_validation_roads,
    grouped_counts_can_meet_minimum,
    maximum_grouped_minimum,
    plan_outer_road_folds,
    select_balanced_evaluation_keys,
)


def make_records(road: str, counts: dict[int, int]):
    rows = []
    tree_id = 0
    for label, count in counts.items():
        for _ in range(count):
            tree_id += 1
            rows.append(
                {
                    "sample_key": f"{road}_1_{tree_id}",
                    "road_id": road,
                    "benchmark_label_id": label,
                }
            )
    return rows


class RoadSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = (1, 2, 3)
        self.instances = []
        road_counts = {
            "00": {1: 12, 2: 1, 3: 1},
            "01": {1: 1, 2: 12, 3: 1},
            "02": {1: 1, 2: 1, 3: 12},
            "03": {1: 8, 2: 8, 3: 8},
            "04": {1: 7, 2: 7, 3: 7},
            "05": {1: 6, 2: 6, 3: 6},
            "06": {1: 5, 2: 5, 3: 5},
            "07": {1: 4, 2: 4, 3: 4},
            "08": {1: 3, 2: 3, 3: 3},
        }
        for road, counts in road_counts.items():
            self.instances.extend(make_records(road, counts))

    def test_outer_folds_are_deterministic_and_road_disjoint(self) -> None:
        counts = build_road_class_counts(self.instances, self.labels)
        first = plan_outer_road_folds(
            counts,
            self.labels,
            n_splits=3,
            minimum_per_class={label: 10 for label in self.labels},
            seed=7,
            restarts=50,
        )
        second = plan_outer_road_folds(
            counts,
            self.labels,
            n_splits=3,
            minimum_per_class={label: 10 for label in self.labels},
            seed=7,
            restarts=50,
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {0, 1, 2})
        for fold in range(3):
            fold_roads = {road for road, assigned in first.items() if assigned == fold}
            for label in self.labels:
                self.assertGreaterEqual(
                    sum(counts[road][label] for road in fold_roads), 10
                )

    def test_inner_validation_retains_training_coverage(self) -> None:
        counts = build_road_class_counts(self.instances, self.labels)
        outer = plan_outer_road_folds(
            counts,
            self.labels,
            n_splits=3,
            minimum_per_class={label: 10 for label in self.labels},
            seed=7,
            restarts=50,
        )
        test_roads = {road for road, fold in outer.items() if fold == 0}
        val_roads = set(
            choose_inner_validation_roads(
                counts,
                self.labels,
                test_roads,
                validation_minimum={label: 5 for label in self.labels},
                training_minimum={label: 5 for label in self.labels},
                target_road_count=2,
                seed=7,
            )
        )
        train_roads = set(counts) - test_roads - val_roads
        self.assertFalse(test_roads & val_roads)
        self.assertFalse(test_roads & train_roads)
        self.assertFalse(val_roads & train_roads)
        for label in self.labels:
            self.assertGreaterEqual(sum(counts[r][label] for r in val_roads), 5)
            self.assertGreaterEqual(sum(counts[r][label] for r in train_roads), 5)

    def test_balanced_selection_is_exact_and_deterministic(self) -> None:
        first = select_balanced_evaluation_keys(
            self.instances,
            self.labels,
            per_class=4,
            seed=19,
            namespace="test",
        )
        second = select_balanced_evaluation_keys(
            self.instances,
            self.labels,
            per_class=4,
            seed=19,
            namespace="test",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)

    def test_insufficient_road_support_is_rejected(self) -> None:
        counts = {
            "00": {1: 10, 2: 10},
            "01": {1: 10, 2: 10},
            "02": {1: 10},
        }
        normalized = {
            road: __import__("collections").Counter(values)
            for road, values in counts.items()
        }
        with self.assertRaisesRegex(ValueError, "appears on 2 roads"):
            plan_outer_road_folds(
                normalized,
                (1, 2),
                n_splits=3,
                minimum_per_class={1: 1, 2: 1},
                seed=3,
                restarts=10,
            )

    def test_exact_grouped_class_capacity(self) -> None:
        self.assertFalse(
            grouped_counts_can_meet_minimum(
                [243, 11, 8], n_splits=3, minimum=10
            )
        )
        self.assertTrue(
            grouped_counts_can_meet_minimum(
                [59, 35, 11, 5, 4, 2, 1],
                n_splits=3,
                minimum=23,
            )
        )
        self.assertEqual(
            maximum_grouped_minimum([243, 11, 8], n_splits=3), 8
        )
        self.assertEqual(
            maximum_grouped_minimum([110, 5, 5, 2], n_splits=3), 5
        )
        self.assertEqual(
            maximum_grouped_minimum([82, 31], n_splits=3), 0
        )


if __name__ == "__main__":
    unittest.main()
