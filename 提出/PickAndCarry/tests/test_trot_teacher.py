"""Unit tests for the open-loop trot teacher (no Genesis)."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.config import CPG_PARAM_DIM, NUM_ACTIONS, layout_slices, ACTION_LAYOUT
from pick_and_carry.cpg import decode_cpg_params
from pick_and_carry.trot_teacher import DEMO_SKILLS, TrotTeacher, skill_command


class TrotTeacherTests(unittest.TestCase):
    def test_action_shape_and_arm_zero(self):
        teacher = TrotTeacher(4, smooth=1.0, device=torch.device("cpu"))
        command = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.1, 0.2], [-0.2, 0.0, 0.0]])
        action = teacher.act(command)
        self.assertEqual(tuple(action.shape), (4, NUM_ACTIONS))
        arm = layout_slices(ACTION_LAYOUT)["arm"]
        grip = layout_slices(ACTION_LAYOUT)["gripper"]
        self.assertTrue(torch.all(action[:, arm] == 0))
        self.assertTrue(torch.all(action[:, grip] == 0))

    def test_standstill_low_amp(self):
        teacher = TrotTeacher(2, smooth=1.0, device=torch.device("cpu"))
        still = torch.zeros(2, 3)
        moving = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
        a_still = teacher.act(still)
        teacher.reset()
        a_move = teacher.act(moving)
        cpg = layout_slices(ACTION_LAYOUT)["cpg"]
        still_amp = float(decode_cpg_params(a_still[:, cpg])["amp"].mean())
        move_amp = float(decode_cpg_params(a_move[:, cpg])["amp"].mean())
        self.assertLess(still_amp, 0.1)
        self.assertGreater(move_amp, 0.3)

    def test_forward_amp_positive(self):
        teacher = TrotTeacher(1, smooth=1.0, device=torch.device("cpu"))
        action = teacher.act(torch.tensor([[0.4, 0.0, 0.0]]))
        cpg = action[:, layout_slices(ACTION_LAYOUT)["cpg"]]
        self.assertGreater(float(cpg[0, 1]), 0.2)

    def test_backward_amp_negative(self):
        teacher = TrotTeacher(1, smooth=1.0, device=torch.device("cpu"))
        action = teacher.act(torch.tensor([[-0.4, 0.0, 0.0]]))
        cpg = action[:, layout_slices(ACTION_LAYOUT)["cpg"]]
        self.assertLess(float(cpg[0, 1]), -0.2)

    def test_sit_uses_pose_bias(self):
        teacher = TrotTeacher(1, smooth=1.0, device=torch.device("cpu"))
        teacher.set_skill("sit")
        teacher._blend.fill_(1.0)
        action = teacher.act(skill_command("sit"), skill="sit")
        cpg = action[:, layout_slices(ACTION_LAYOUT)["cpg"]]
        decoded = decode_cpg_params(cpg)
        self.assertLess(float(decoded["amp"][0]), 0.1)
        self.assertGreater(float(decoded["bias_th"][0]), 0.3)

    def test_jump_crouch_then_extend(self):
        teacher = TrotTeacher(1, dt=0.02, smooth=1.0, device=torch.device("cpu"))
        teacher.set_skill("jump")
        teacher._blend.fill_(1.0)
        mid = None
        for _ in range(15):
            mid = teacher.act(skill_command("jump"), skill="jump")
        extend = mid
        for _ in range(15):
            extend = teacher.act(skill_command("jump"), skill="jump")
        cpg = layout_slices(ACTION_LAYOUT)["cpg"]
        mid_th = float(decode_cpg_params(mid[:, cpg])["bias_th"][0])
        ext_th = float(decode_cpg_params(extend[:, cpg])["bias_th"][0])
        self.assertGreater(mid_th, 0.3)
        self.assertLess(ext_th, mid_th)

    def test_turn_uses_yaw_command(self):
        left = skill_command("turn_left")
        right = skill_command("turn_right")
        self.assertGreater(float(left[0, 2]), 0.4)
        self.assertLess(float(right[0, 2]), -0.4)

    def test_turn_left_sets_positive_turn_param(self):
        teacher = TrotTeacher(1, smooth=1.0, device=torch.device("cpu"))
        teacher.set_skill("turn_left")
        teacher._blend.fill_(1.0)
        action = teacher.act(skill_command("turn_left"), skill="turn_left")
        cpg = action[:, layout_slices(ACTION_LAYOUT)["cpg"]]
        self.assertGreater(float(cpg[0, 7]), 0.3)
        self.assertEqual(cpg.shape[-1], CPG_PARAM_DIM)

    def test_demo_skills_complete(self):
        self.assertEqual(
            set(DEMO_SKILLS),
            {"stand", "sit", "jump", "forward", "backward", "turn_left", "turn_right", "stop"},
        )


if __name__ == "__main__":
    unittest.main()
