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
    migrate_legacy_imitation_videos,
    video_family_for_kind,
    video_task_name,
    videos_root,
)


class VideoPathTests(unittest.TestCase):
    def test_task_names(self):
        self.assertEqual(video_task_name("walk", easy=True), "walk_easy")
        self.assertEqual(video_task_name("walk", easy=False), "walk")
        self.assertEqual(video_task_name("catch", catch_level="fly"), "catch_fly")
        self.assertEqual(video_task_name("carry"), "carry")

    def test_imitation_kinds_use_imitation_family(self):
        self.assertEqual(video_family_for_kind("teacher"), "imitation")
        self.assertEqual(video_family_for_kind("bc"), "imitation")
        self.assertEqual(video_family_for_kind("dagger"), "imitation")
        self.assertIsNone(video_family_for_kind("ppo"))
        self.assertIsNone(video_family_for_kind("grpo"))

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
            self.assertEqual(first.parent.parent.name, "imitation")
            self.assertEqual(first.parent.parent.parent.name, "videos")
            (first / "trot_teacher_demo.mp4").write_bytes(b"fake")
            second = make_video_run_dir(
                root, task="walk_easy", kind="bc", epochs=80, stamp="20261018-110000"
            )
            (second / "bc_trot_demo.mp4").write_bytes(b"fake")
            ppo = make_video_run_dir(root, task="walk_easy", kind="ppo", iteration=10, stamp="20261018-120000")
            self.assertEqual(ppo.parent.parent.name, "videos")
            self.assertNotEqual(ppo.parent.parent.name, "imitation")
            self.assertEqual(latest_video_run(root, task="walk_easy", kind="bc"), second)
            self.assertEqual(find_named_mp4(root, "bc_trot_demo.mp4"), second / "bc_trot_demo.mp4")
            self.assertTrue((second / "run.json").is_file())
            self.assertEqual(videos_root(root, family="imitation"), root / "videos" / "imitation")

    def test_migrate_flat_imitation_runs_under_videos_imitation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "videos" / "walk_easy" / "bc_ep0080_20261018-100000"
            flat.mkdir(parents=True)
            (flat / "bc_trot_demo.mp4").write_bytes(b"fake")
            ppo = root / "videos" / "walk_easy" / "ppo_iter000010_20261018-110000"
            ppo.mkdir(parents=True)
            (ppo / "play.mp4").write_bytes(b"fake")
            moved = migrate_legacy_imitation_videos(root)
            self.assertEqual(len(moved), 1)
            dest = root / "videos" / "imitation" / "walk_easy" / "bc_ep0080_20261018-100000"
            self.assertTrue((dest / "bc_trot_demo.mp4").is_file())
            self.assertFalse(flat.exists())
            self.assertTrue((ppo / "play.mp4").is_file())


if __name__ == "__main__":
    unittest.main()
