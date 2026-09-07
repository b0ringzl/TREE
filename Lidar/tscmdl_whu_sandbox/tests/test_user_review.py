from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.user_review import (  # noqa: E402
    REVIEW_SCOPE_ALL,
    complete_scope,
    export_action_lists,
    inspect_user_review_state,
    load_or_create_review_state,
    refresh_validation_review_status,
    save_user_review,
    validate_review_payload,
)


def record(key: str, quality: bool = False) -> dict[str, object]:
    return {
        "sample_key": key,
        "class_index": 0,
        "scientific_name": "Cinnamomum camphora",
        "benchmark_split": "train",
        "road_id": "01",
        "trajectory_id": "1",
        "tree_id": int(key[-1]),
        "point_path": f"assets/points/{key}.npz",
        "image_path": f"assets/images/{key}.jpg",
        "quality_preview_path": f"quality/{key}.jpg" if quality else "",
    }


class UserReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = [record("sample1", True), record("sample2")]
        self.manifest = self.root / "shared_manifest.json"
        self.manifest.write_text(
            json.dumps({"records": self.records}), encoding="utf-8"
        )
        self.review_path = self.root / "user_visual_review.json"
        self.state = load_or_create_review_state(
            self.review_path, self.root, self.manifest
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rating_and_action_rules_prevent_contradictions(self) -> None:
        with self.assertRaises(ValueError):
            validate_review_payload(
                {"sample_key": "sample1", "rating": "pass", "action": "exclude_image"}
            )
        with self.assertRaises(ValueError):
            validate_review_payload(
                {"sample_key": "sample1", "rating": "reject", "action": "keep"}
            )

    def test_scope_is_not_complete_until_every_sample_is_reviewed(self) -> None:
        save_user_review(
            self.review_path,
            self.state,
            {"sample_key": "sample1", "rating": "pass", "action": "keep"},
            {"sample1", "sample2"},
        )
        status = complete_scope(
            self.review_path, self.state, self.records, REVIEW_SCOPE_ALL
        )
        self.assertEqual(status["status"], "in_progress")
        self.assertEqual(status["remaining"], 1)
        inspected = inspect_user_review_state(
            self.review_path, self.manifest, self.records, REVIEW_SCOPE_ALL
        )
        self.assertEqual(inspected["status"], "in_progress")

    def test_formal_completion_requires_user_reviews_and_explicit_close(self) -> None:
        for key in ("sample1", "sample2"):
            save_user_review(
                self.review_path,
                self.state,
                {"sample_key": key, "rating": "pass", "action": "keep"},
                {"sample1", "sample2"},
            )
        before_close = inspect_user_review_state(
            self.review_path, self.manifest, self.records, REVIEW_SCOPE_ALL
        )
        self.assertEqual(before_close["status"], "in_progress")
        complete_scope(self.review_path, self.state, self.records, REVIEW_SCOPE_ALL)
        after_close = inspect_user_review_state(
            self.review_path, self.manifest, self.records, REVIEW_SCOPE_ALL
        )
        self.assertEqual(after_close["status"], "complete")

    def test_export_is_non_destructive_and_separates_actions(self) -> None:
        for payload in (
            {"sample_key": "sample1", "rating": "pass", "action": "keep"},
            {
                "sample_key": "sample2",
                "rating": "reject",
                "action": "exclude_sample",
                "reasons": ["target_missing"],
            },
        ):
            save_user_review(
                self.review_path,
                self.state,
                payload,
                {"sample1", "sample2"},
            )
        before = self.manifest.read_bytes()
        summary = export_action_lists(
            self.root / "review_outputs", self.records, self.state
        )
        self.assertEqual(summary["accepted_multimodal_count"], 1)
        self.assertEqual(summary["excluded_sample_count"], 1)
        self.assertEqual(summary["multimodal_training_count"], 1)
        self.assertEqual(summary["point_training_count"], 1)
        self.assertEqual(summary["multimodal_quarantine_count"], 1)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertTrue((self.root / "review_outputs" / "excluded_samples.csv").is_file())

    def test_validation_status_refresh_keeps_technical_result(self) -> None:
        validation_path = self.root / "validation.json"
        validation_path.write_text(
            json.dumps(
                {
                    "stage": "C1",
                    "status": "passed",
                    "review_status": {"codex_visual_precheck": "complete"},
                    "summary": {},
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )
        review_status = inspect_user_review_state(
            self.review_path, self.manifest, self.records, REVIEW_SCOPE_ALL
        )
        refresh_validation_review_status(
            validation_path, self.review_path, review_status
        )
        refreshed = json.loads(validation_path.read_text(encoding="utf-8"))
        self.assertEqual(refreshed["status"], "passed")
        self.assertEqual(
            refreshed["review_status"]["user_visual_review"]["status"],
            "pending",
        )
        self.assertFalse(refreshed["summary"]["user_visual_review_complete"])


if __name__ == "__main__":
    unittest.main()
