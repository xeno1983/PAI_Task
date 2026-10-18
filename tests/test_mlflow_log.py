"""Local MLflow records. No simulator."""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import tempfile
import unittest
from pathlib import Path

from pick_and_carry.mlflow_log import LocalTracker, release_store


class LocalTrackerTests(unittest.TestCase):
    def test_a_run_keeps_the_parent_checkpoint_and_the_curve(self):
        import mlflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs" / "ppo_curriculum_walk"
            log_dir.mkdir(parents=True)
            checkpoint = log_dir / "ppo_0020.pt"
            checkpoint.write_bytes(b"weights")
            tracker = LocalTracker()
            try:
                tracker.start(
                    log_dir,
                    {
                        "algorithm": "ppo",
                        "curriculum": "walk",
                        "catch_level": "static",
                        "arm": "z1",
                        "obs_dim": 69,
                        "act_dim": 17,
                        "num_envs": 16,
                        "start_iteration": 0,
                        "resumed_from": str(log_dir.parent / "ppo_curriculum_sit" / "ppo_0100.pt"),
                    },
                )
                self.assertTrue(tracker.active)
                tracker.log_row({"iteration": 20, "reward": 1.5, "tracking": 0.4})
                tracker.log_checkpoint(checkpoint)
                tracker.close()
                self.assertFalse(tracker.active)

                mlflow.set_tracking_uri("sqlite:///" + (root / "logs" / "mlflow" / "mlflow.db").resolve().as_posix())
                experiment = mlflow.get_experiment_by_name("pick-and-carry")
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs.iloc[0]["params.curriculum"], "walk")
                self.assertEqual(runs.iloc[0]["params.obs_dim"], "69")
                self.assertEqual(runs.iloc[0]["params.act_dim"], "17")
                self.assertIn("ppo_0100.pt", runs.iloc[0]["tags.resumed_from"])
                self.assertAlmostEqual(float(runs.iloc[0]["metrics.reward"]), 1.5)
                artifacts = mlflow.artifacts.list_artifacts(
                    run_id=runs.iloc[0]["run_id"], artifact_path="checkpoints"
                )
                self.assertEqual([item.path for item in artifacts], ["checkpoints/ppo_0020.pt"])
            finally:
                release_store()
