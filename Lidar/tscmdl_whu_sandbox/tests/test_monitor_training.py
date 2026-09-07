from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "scripts"))

import monitor_training  # noqa: E402


class TrainingMonitorTests(unittest.TestCase):
    def test_completed_smoke_run_is_not_reported_as_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run_config.json").write_text(
                json.dumps({"args": {"epochs": 1}}), encoding="utf-8"
            )
            (run_dir / "run_state.json").write_text(
                json.dumps({"status": "smoke_complete"}), encoding="utf-8"
            )
            (run_dir / "smoke_result.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            with patch.object(monitor_training, "gpu_snapshot", return_value=None):
                snapshot = monitor_training.build_snapshot(run_dir)

            self.assertEqual(snapshot["state"], "smoke_complete")
            self.assertEqual(snapshot["smoke"]["status"], "passed")
            self.assertEqual(snapshot["progress_fraction"], 1.0)
            self.assertEqual(snapshot["eta_seconds"], 0.0)

    def test_live_batch_progress_is_reported_before_first_epoch_finishes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "model": "PTv2",
                        "parameter_count": 100,
                        "split_sizes": {
                            "train": 1000,
                            "val": 100,
                            "test": 100,
                        },
                        "args": {
                            "epochs": 10,
                            "batch_size": 16,
                            "eval_batch_size": 16,
                            "learning_rate": 0.001,
                            "patience": 2,
                            "seed": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "live_progress.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "phase": "train",
                        "epoch": 1,
                        "total_epochs": 10,
                        "batch": 25,
                        "total_batches": 100,
                        "loss": 2.5,
                        "accuracy": 0.25,
                        "macro_f1": 0.2,
                        "phase_eta_seconds": 75,
                        "peak_allocated_mib": 4096,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(monitor_training, "gpu_snapshot", return_value=None):
                snapshot = monitor_training.build_snapshot(run_dir)

            self.assertEqual(snapshot["state"], "training")
            self.assertEqual(snapshot["epoch"], 1)
            self.assertAlmostEqual(snapshot["progress_fraction"], 0.0225)
            rendered = monitor_training.render_text(snapshot)
            self.assertIn("25/100", rendered)
            self.assertIn("loss=2.5000", rendered)
            self.assertIn("macro_f1=20.00%", rendered)


if __name__ == "__main__":
    unittest.main()
