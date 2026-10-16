"""Socket bridge used by inference. No ROS and no simulator."""

from __future__ import annotations

import base64
import json
import socket
import time
import unittest

import numpy as np

from pick_and_carry.ros_bridge import (
    InferenceBridge,
    clamp_goal_xy,
    encode_message,
    goal_from_message,
    pack_rgb_image,
    quat_wxyz_to_xyzw,
    ros_usage_error,
    split_messages,
)
from pick_and_carry.terrain import COURSE_X, COURSE_Y


class GoalClampTests(unittest.TestCase):
    def test_inside_course_is_unchanged(self):
        self.assertEqual(clamp_goal_xy(1.5, -0.2), (1.5, -0.2))

    def test_outside_course_is_clamped(self):
        self.assertEqual(clamp_goal_xy(99.0, -99.0), (COURSE_X[1], COURSE_Y[0]))


class ProtocolTests(unittest.TestCase):
    def test_quaternion_order(self):
        self.assertEqual(quat_wxyz_to_xyzw([1.0, 0.0, 0.2, 0.3]), [0.0, 0.2, 0.3, 1.0])

    def test_partial_line_waits(self):
        first, rest = split_messages(b'{"type":"goal","x":1,"y":2}\n{"type":')
        self.assertEqual(first, [{"type": "goal", "x": 1, "y": 2}])
        self.assertEqual(rest, b'{"type":')

    def test_bad_line_is_skipped(self):
        messages, rest = split_messages(b"not-json\n" + encode_message({"type": "goal", "x": 1.0, "y": 0.0}))
        self.assertEqual(goal_from_message(messages[0]), (1.0, 0.0))
        self.assertEqual(rest, b"")

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
        from argparse import Namespace

        from pick_and_carry.train_ppo import make_config

        args = Namespace(
            terrain_amplitude=None,
            num_envs=1,
            backend="cpu",
            viewer=False,
            record=None,
            control="position",
            no_assist_grasp=False,
            freeze_legs=False,
            scripted_gait=False,
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


class BridgeTests(unittest.TestCase):
    def test_latest_goal_and_state_line(self):
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(2.0)
        bridge = InferenceBridge("127.0.0.1", server.getsockname()[1])
        bridge.start()
        client = None
        try:
            client, _ = server.accept()
            client.settimeout(2.0)
            client.sendall(encode_message({"type": "goal", "x": 100.0, "y": 0.0}))
            client.sendall(b"not-json\n")
            client.sendall(encode_message({"type": "goal", "x": 1.5, "y": 0.25}))
            goal = None
            for _ in range(50):
                goal = bridge.poll_goal()
                if goal == (1.5, 0.25):
                    break
                time.sleep(0.02)
            self.assertEqual(goal, (1.5, 0.25))
            self.assertIsNone(bridge.poll_goal())

            bridge.publish({"type": "state", "phase": 2, "arm": "z1"})
            data = b""
            while b"\n" not in data:
                data += client.recv(4096)
            payload = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
            self.assertEqual(payload["phase"], 2)
            self.assertEqual(payload["arm"], "z1")
        finally:
            if client is not None:
                client.close()
            server.close()
            bridge.close()


class FileBridgeTests(unittest.TestCase):
    def test_goal_sequence_and_state_without_image(self):
        import tempfile
        from pathlib import Path

        from pick_and_carry.ros_bridge import FileBridge

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
