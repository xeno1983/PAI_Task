"""Unit tests for Z1 FK / IK (no Genesis)."""

from __future__ import annotations

import math
import unittest

import torch

from pick_and_carry.arms import get_arm
from pick_and_carry.ik import (
    forward_kinematics,
    forward_kinematics_and_jacobian,
    numerical_jacobian,
    solve_ik,
)


class IkTests(unittest.TestCase):
    def setUp(self):
        arm = get_arm("z1")
        self.default = torch.tensor([arm.default_angles[n] for n in arm.joints], dtype=torch.float32)
        self.lower = torch.tensor([arm.limits[n][0] for n in arm.joints], dtype=torch.float32)
        self.upper = torch.tensor([arm.limits[n][1] for n in arm.joints], dtype=torch.float32)

    def test_default_pose_is_reachable_ahead(self):
        pos = forward_kinematics(self.default)
        # Mount frame: roughly forward (+X-ish after shoulder pitch) and above mount.
        self.assertGreater(float(pos[0, 0].abs() + pos[0, 2].abs()), 0.15)

    def test_ik_recovers_nearby_target(self):
        q_true = self.default.clone()
        q_true[0] += 0.2
        q_true[1] -= 0.15
        q_true = q_true.clamp(self.lower, self.upper)
        target = forward_kinematics(q_true)
        q_ik, err = solve_ik(self.default, target, self.lower, self.upper)
        self.assertLess(float(err[0]), 0.02)
        pos = forward_kinematics(q_ik)
        self.assertLess(float((pos - target).norm()), 0.02)

    def test_ik_stays_in_limits(self):
        target = forward_kinematics(self.default) + torch.tensor([[0.05, 0.03, -0.02]])
        q_ik, _ = solve_ik(self.default, target, self.lower, self.upper)
        self.assertTrue(torch.all(q_ik >= self.lower - 1e-5))
        self.assertTrue(torch.all(q_ik <= self.upper + 1e-5))

    def test_batched(self):
        q0 = self.default.expand(4, -1).clone()
        target = forward_kinematics(q0)
        target[:, 0] += torch.linspace(-0.04, 0.04, 4)
        q_ik, err = solve_ik(q0, target, self.lower, self.upper)
        self.assertEqual(tuple(q_ik.shape), (4, 6))
        self.assertEqual(tuple(err.shape), (4,))
        self.assertLess(float(err.max()), 0.03)

    def test_analytic_jacobian_matches_numeric(self):
        q = self.default.unsqueeze(0).clone()
        q[0, 0] += 0.3
        q[0, 2] -= 0.2
        q[0, 4] += 0.1
        _, jac_a = forward_kinematics_and_jacobian(q)
        jac_n = numerical_jacobian(q)
        self.assertLess(float((jac_a - jac_n).abs().max()), 2e-3)


if __name__ == "__main__":
    unittest.main()
