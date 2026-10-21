"""Learning-curve records. No simulator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pick_and_carry.tracking import append_metrics, load_metrics, render_curves, write_run


class MetricsTests(unittest.TestCase):
    def test_later_line_replaces_the_same_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            append_metrics(path, {"iteration": 1, "reward": 0.1})
            append_metrics(path, {"iteration": 2, "reward": 0.2})
            append_metrics(path, {"iteration": 1, "reward": 0.5})
            rows = load_metrics(path)
        self.assertEqual([row["iteration"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["reward"], 0.5)

    def test_run_file_is_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_run(path, {"algorithm": "ppo", "num_envs": 16})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["algorithm"], "ppo")
        self.assertIn("recorded_at", payload)


class CurvePageTests(unittest.TestCase):
    def test_page_contains_the_series(self):
        rows = [
            {"iteration": 1, "reward": 0.1, "grasp": 0.0, "ee_dist": 0.8},
            {"iteration": 2, "reward": 0.4, "grasp": 0.1, "ee_dist": 0.5, "success": 0.0},
        ]
        page = render_curves(rows, {"num_envs": 16})
        self.assertIn("報酬", page)
        self.assertIn("把持率", page)
        self.assertIn("設置率", page)
        self.assertIn('class="trend"', page)
        self.assertIn("num_envs", page)

    def test_empty_page_explains_itself(self):
        page = render_curves([])
        self.assertIn("学習記録がまだありません", page)


if __name__ == "__main__":
    unittest.main()
