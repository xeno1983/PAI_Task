"""Arm / gripper residual command helpers."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.control import (
    arm_residual_command,
    gripper_baseline,
    gripper_residual_command,
    arm_gripper_command,
)


class ResidualControlTests(unittest.TestCase):
    def test_zero_arm_residual_matches_ik(self):
        q_ik = torch.zeros(2, 6)
        q_ik[:, 1] = 1.5
        lower = torch.full((6,), -3.0)
        upper = torch.full((6,), 3.0)
        out = arm_residual_command(q_ik, torch.zeros(2, 6), arm_scale=0.12, lower=lower, upper=upper)
        self.assertTrue(torch.allclose(out, q_ik))

    def test_arm_residual_scaled(self):
        q_ik = torch.zeros(1, 6)
        act = torch.ones(1, 6)
        lower = torch.full((6,), -3.0)
        upper = torch.full((6,), 3.0)
        out = arm_residual_command(q_ik, act, arm_scale=0.12, lower=lower, upper=upper)
        self.assertTrue(torch.allclose(out, torch.full((1, 6), 0.12)))

    def test_gripper_baseline_phases(self):
        phase = torch.tensor([0, 1, 2, 3])
        base = gripper_baseline(phase, curriculum="carry", gripper_open=-1.57, gripper_closed=0.0)
        self.assertAlmostEqual(float(base[0]), -1.57, places=2)
        self.assertAlmostEqual(float(base[2]), 0.0, places=2)
        self.assertAlmostEqual(float(base[3]), -1.57, places=2)

    def test_ik_path_zero_residual(self):
        q_ik = torch.tensor([[0.0, 1.0, -0.5, 0.0, 0.0, 0.0]])
        default = torch.zeros(1, 6)
        lower = torch.full((6,), -3.0)
        upper = torch.full((6,), 3.0)
        arm, grip = arm_gripper_command(
            torch.zeros(1, 6),
            torch.zeros(1, 1),
            default,
            torch.tensor([2]),
            control_split="both",
            arm_scale=0.12,
            gripper_open=-1.57,
            gripper_closed=0.0,
            lower=lower,
            upper=upper,
            q_ik=q_ik,
            grip_residual_scale=0.35,
            curriculum="carry",
        )
        self.assertTrue(torch.allclose(arm, q_ik))
        self.assertAlmostEqual(float(grip[0]), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
