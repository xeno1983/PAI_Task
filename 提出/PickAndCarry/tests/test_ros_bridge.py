"""File bridge used by inference. No ROS and no simulator."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from pick_and_carry.ros_bridge import (
    FileBridge,
    clamp_goal_xy,
    goal_from_message,
    pack_rgb_image,
    ros_usage_error,
)
from pick_and_carry.terrain import COURSE_X, COURSE_Y
from pick_and_carry.train_ppo import make_config


class GoalClampTests(unittest.TestCase):
    def test_inside_course_is_unchanged(self):
        self.assertEqual(clamp_goal_xy(1.5, -0.2), (1.5, -0.2))

    def test_outside_course_is_clamped(self):
        self.assertEqual(clamp_goal_xy(99.0, -99.0), (COURSE_X[1], COURSE_Y[0]))


class ProtocolTests(unittest.TestCase):
    def test_goal_message(self):
        self.assertEqual(goal_from_message({"type": "goal", "x": 1.0, "y": 0.0}), (1.0, 0.0))
        self.assertIsNone(goal_from_message({"type": "state"}))

    def test_overview_image_is_downsampled_rgb(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[0, 0] = (1, 2, 3)
        packed = pack_rgb_image(frame, stride=2)
        raw = base64.b64decode(packed["data"])
        self.assertEqual((packed["w"], packed["h"]), (3, 2))
        self.assertEqual(len(raw), 3 * 2 * 3)
        self.assertEqual(raw[:3], bytes((1, 2, 3)))

    def test_ros_is_play_only(self):
        self.assertIsNone(ros_usage_error(enabled=False, play=False, num_envs=16))
        self.assertIn("--play", ros_usage_error(enabled=True, play=False, num_envs=1))
        self.assertIn("--num-envs 1", ros_usage_error(enabled=True, play=True, num_envs=4))
        self.assertIsNone(ros_usage_error(enabled=True, play=True, num_envs=1))

    def test_ros_play_covers_the_whole_recording(self):
        args = Namespace(
            terrain_amplitude=None,
            num_envs=1,
            backend="cpu",
            viewer=False,
            record=None,
            seed=0,
            terrain_steps=False,
            control_split="split",
            arm="z1",
            wrist_camera=False,
            object_xy=None,
            goal_xy=None,
            play=True,
            ros=True,
            seconds=60.0,
        )
        self.assertGreaterEqual(make_config(args).episode_length_s, 60.0)


class FileBridgeTests(unittest.TestCase):
    def test_goal_sequence_and_state_without_image(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = FileBridge(folder)
            (Path(folder) / "goal-1.json").write_text(
                json.dumps({"type": "goal", "x": 100.0, "y": 0.2, "seq": 1}), encoding="utf-8"
            )
            self.assertEqual(bridge.poll_goal(), (COURSE_X[1], 0.2))
            self.assertIsNone(bridge.poll_goal())
            (Path(folder) / "goal-2.json").write_text(
                json.dumps({"type": "goal", "x": 1.5, "y": -0.2, "seq": 2}), encoding="utf-8"
            )
            self.assertEqual(bridge.poll_goal(), (1.5, -0.2))
            bridge.publish({"type": "state", "phase": 1, "image": {"data": "skip"}})
            state = json.loads((Path(folder) / "state-1.json").read_text(encoding="utf-8"))
            self.assertEqual(state["phase"], 1)
            self.assertEqual(state["seq"], 1)
            self.assertNotIn("image", state)
            held = (Path(folder) / "state-1.json").open("rb")
            try:
                bridge.publish({"type": "state", "phase": 2})
            finally:
                held.close()
            self.assertTrue((Path(folder) / "state-2.json").is_file())
