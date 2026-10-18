"""Unit tests for the open-loop trot teacher (no Genesis)."""

from __future__ import annotations

import math
import unittest

import torch

from pick_and_carry.config import NUM_ACTIONS
from pick_and_carry.trot_teacher import TrotTeacher


class TrotTeacherTests(unittest.TestCase):
    def test_action_shape_and_arm_zero(self):
        teacher = TrotTeacher(4, device=torch.device("cpu"))
        command = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.1, 0.2], [-0.2, 0.0, 0.0]])
        action = teacher.act(command)
        self.assertEqual(tuple(action.shape), (4, NUM_ACTIONS))
        self.assertTrue(torch.all(action[:, 12:] == 0))

    def test_standstill_near_zero_legs(self):
        teacher = TrotTeacher(2, thigh_amp=0.4, calf_amp=0.5, device=torch.device("cpu"))
        still = torch.zeros(2, 3)
        moving = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
        a_still = teacher.act(still)
        teacher.reset()
        a_move = teacher.act(moving)
        self.assertLess(float(a_still[:, :12].abs().mean()), 1e-5)
        self.assertGreater(float(a_move[:, :12].abs().mean()), 0.05)

    def test_backward_flips_leg_swing_sign(self):
        teacher = TrotTeacher(1, thigh_amp=0.4, calf_amp=0.5, device=torch.device("cpu"))
        teacher.phase.fill_(math.pi / 2)  # sin(phase)=1 so swing sign is obvious
        forward = teacher.act(torch.tensor([[0.4, 0.0, 0.0]]))
        teacher.phase.fill_(math.pi / 2)
        backward = teacher.act(torch.tensor([[-0.4, 0.0, 0.0]]))
        # With _SAGITTAL_SIGN=-1, +vx yields negative FL thigh action at this phase.
        self.assertLess(float(forward[0, 1]), -0.05)
        self.assertGreater(float(backward[0, 1]), 0.05)
        self.assertAlmostEqual(float(forward[0, 1]), -float(backward[0, 1]), places=4)

    def test_phase_advances(self):
        teacher = TrotTeacher(1, frequency_hz=2.0, dt=0.02, device=torch.device("cpu"))
        cmd = torch.tensor([[0.4, 0.0, 0.0]])
        p0 = float(teacher.phase[0])
        teacher.act(cmd)
        p1 = float(teacher.phase[0])
        self.assertNotAlmostEqual(p0, p1, places=5)


if __name__ == "__main__":
    unittest.main()
