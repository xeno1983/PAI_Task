"""The arm is the Unitree Z1: observation 69, action 19."""

from __future__ import annotations

import math
import unittest
import xml.etree.ElementTree as ET

import torch

from pick_and_carry.arms import get_arm
from pick_and_carry.config import NUM_ACTIONS, NUM_OBS
from pick_and_carry.control import arm_gripper_command


class ArmLayoutTests(unittest.TestCase):
    def test_only_the_z1_is_available(self):
        z1 = get_arm("z1")
        self.assertEqual(z1.num_obs, 69)
        self.assertEqual(z1.num_actions, 19)
        self.assertEqual(NUM_OBS, 69)
        self.assertEqual(NUM_ACTIONS, 19)
        with self.assertRaises(ValueError):
            get_arm("panda")

    def test_z1_is_under_eight_kilograms_and_has_six_joints(self):
        z1 = get_arm("z1")
        self.assertEqual(z1.dof, 6)
        self.assertEqual(z1.num_obs, 69)
        self.assertEqual(z1.num_actions, 19)
        root = ET.parse(z1.urdf_path).getroot()
        mass = 0.0
        for link in root.findall("link"):
            if link.get("name", "").startswith("z1_"):
                mass += float(link.find("inertial/mass").get("value"))
        self.assertGreater(mass, 4.0)
        self.assertLess(mass, 8.0)
        for name, angle in z1.default_angles.items():
            lower, upper, _effort = z1.limits[name]
            self.assertGreaterEqual(angle, lower)
            self.assertLessEqual(angle, upper)

    def test_z1_gripper_closes_at_action_minus_one(self):
        z1 = get_arm("z1")
        _arm, opening = arm_gripper_command(
            torch.zeros(1, 6),
            torch.tensor([[-1.0]]),
            torch.zeros(1, 6),
            torch.tensor([1]),
            control_split="both",
            arm_scale=1.0,
            gripper_open=z1.joint_open,
            gripper_closed=z1.joint_closed,
            lower=torch.full((6,), -3.0),
            upper=torch.full((6,), 3.0),
        )
        self.assertAlmostEqual(float(opening), 0.0)
        _arm, opened = arm_gripper_command(
            torch.zeros(1, 6),
            torch.tensor([[1.0]]),
            torch.zeros(1, 6),
            torch.tensor([1]),
            control_split="both",
            arm_scale=1.0,
            gripper_open=z1.joint_open,
            gripper_closed=z1.joint_closed,
            lower=torch.full((6,), -3.0),
            upper=torch.full((6,), 3.0),
        )
        self.assertAlmostEqual(float(opened), -0.5 * math.pi)


if __name__ == "__main__":
    unittest.main()
