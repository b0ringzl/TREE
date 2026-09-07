from __future__ import annotations

import sys
import unittest
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.status import (  # noqa: E402
    PredictionRecord,
    compare_prediction_models,
    compare_prediction_records,
)


class StatusComparisonTests(unittest.TestCase):
    def test_prediction_complementarity_counts(self) -> None:
        left = {
            "a": PredictionRecord("a", "test", 0, 0),
            "b": PredictionRecord("b", "test", 1, 1),
            "c": PredictionRecord("c", "test", 2, 0),
            "d": PredictionRecord("d", "test", 0, 2),
        }
        right = {
            "a": PredictionRecord("a", "test", 0, 0),
            "b": PredictionRecord("b", "test", 1, 2),
            "c": PredictionRecord("c", "test", 2, 2),
            "d": PredictionRecord("d", "test", 0, 1),
        }

        result = compare_prediction_records(left, right, 3)
        summary = result["by_split"]["test"]
        self.assertEqual(summary["both_correct"], 1)
        self.assertEqual(summary["left_only_correct"], 1)
        self.assertEqual(summary["right_only_correct"], 1)
        self.assertEqual(summary["neither_correct"], 1)
        self.assertEqual(summary["same_prediction"], 1)
        self.assertAlmostEqual(summary["either_model_oracle_accuracy"], 0.75)

    def test_prediction_metadata_mismatch_is_rejected(self) -> None:
        left = {"a": PredictionRecord("a", "val", 0, 0)}
        right = {"a": PredictionRecord("a", "test", 0, 0)}
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            compare_prediction_records(left, right, 3)

    def test_three_model_overlap_patterns(self) -> None:
        models = {
            "B2": {
                "a": PredictionRecord("a", "test", 0, 0),
                "b": PredictionRecord("b", "test", 1, 0),
            },
            "B3": {
                "a": PredictionRecord("a", "test", 0, 1),
                "b": PredictionRecord("b", "test", 1, 1),
            },
            "B4": {
                "a": PredictionRecord("a", "test", 0, 0),
                "b": PredictionRecord("b", "test", 1, 1),
            },
        }
        result = compare_prediction_models(models, 3)["by_split"]["test"]
        self.assertEqual(result["pattern_counts"], {"101": 1, "011": 1})
        self.assertEqual(result["correct_by_model"], {"B2": 1, "B3": 1, "B4": 2})
        self.assertEqual(result["all_correct"], 0)
        self.assertEqual(result["none_correct"], 0)
        self.assertEqual(result["any_model_oracle_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
