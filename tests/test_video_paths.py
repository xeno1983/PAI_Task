"""Video subfolder naming helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pick_and_carry.video_paths import (
    find_named_mp4,
    latest_video_run,
    make_run_id,
    make_video_run_dir,
    video_task_name,
)


class VideoPathTests(unittest.TestCase):
    def test_task_names(self):
        self.assertEqual(video_task_name("walk", easy=True), "walk_easy")
        self.assertEqual(video_task_name("walk", easy=False), "walk")
        self.assertEqual(video_task_name("catch", catch_level="fly"), "catch_fly")
        self.assertEqual(video_task_name("carry"), "carry")

    def test_run_id_includes_kind_and_progress(self):
        run = make_run_id("bc", epochs=80, stamp="20261018-120000")
        self.assertTrue(run.startswith("bc_ep0080_"))
        self.assertIn("20261018-120000", run)
        ppo = make_run_id("ppo", iteration=2000, stamp="20261018-120000")
        self.assertIn("iter002000", ppo)

    def test_make_and_find_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_video_run_dir(root, task="walk_easy", kind="teacher", stamp="20261018-100000")
            (first / "trot_teacher_demo.mp4").write_bytes(b"fake")
            second = make_video_run_dir(
                root, task="walk_easy", kind="bc", epochs=80, stamp="20261018-110000"
            )
            (second / "bc_trot_demo.mp4").write_bytes(b"fake")
            self.assertEqual(latest_video_run(root, task="walk_easy", kind="bc"), second)
            self.assertEqual(find_named_mp4(root, "bc_trot_demo.mp4"), second / "bc_trot_demo.mp4")
            self.assertTrue((second / "run.json").is_file())


if __name__ == "__main__":
    unittest.main()
