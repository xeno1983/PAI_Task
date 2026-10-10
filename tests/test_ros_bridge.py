"""Socket bridge used by inference. No ROS and no simulator."""

from __future__ import annotations

import json
import socket
import time
import unittest

from pick_and_carry.ros_bridge import (
    InferenceBridge,
    clamp_goal_xy,
    encode_message,
    goal_from_message,
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
            arm="panda",
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
