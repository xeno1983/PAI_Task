"""DAgger helper unit tests (no simulator)."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.dagger import action_disagreement, dagger_beta, should_query_teacher
from pick_and_carry.trot_teacher import TrotTeacher


class DaggerHelperTests(unittest.TestCase):
    def test_beta_decays_and_clamps(self):
        self.assertAlmostEqual(dagger_beta(1, beta0=1.0, decay=0.5), 1.0)
        self.assertAlmostEqual(dagger_beta(2, beta0=1.0, decay=0.5), 0.5)
        self.assertAlmostEqual(dagger_beta(3, beta0=1.0, decay=0.5), 0.25)
        self.assertGreaterEqual(dagger_beta(100, beta0=1.0, decay=0.5), 0.0)

    def test_classic_always_queries(self):
        d = torch.tensor([0.0, 0.01, 1.0])
        q = should_query_teacher(d, threshold=0.5, mode="classic")
        self.assertTrue(bool(q.all()))

    def test_gated_uses_threshold(self):
        d = torch.tensor([0.0, 0.2, 0.5])
        q = should_query_teacher(d, threshold=0.25, mode="gated")
        self.assertEqual(q.tolist(), [False, False, True])

    def test_disagreement_respects_mask(self):
        policy = torch.tensor([[1.0, 2.0, 3.0]])
        teacher = torch.tensor([[1.0, 2.0, 0.0]])
        mask = torch.tensor([1.0, 1.0, 0.0])
        d_masked = action_disagreement(policy, teacher, mask)
        d_full = action_disagreement(policy, teacher, None)
        self.assertAlmostEqual(float(d_masked), 0.0, places=5)
        self.assertGreater(float(d_full), 0.0)


class TeacherAdvanceTests(unittest.TestCase):
    def test_peek_does_not_advance_blend(self):
        teacher = TrotTeacher(num_envs=2, device=torch.device("cpu"))
        teacher.set_skill("forward")
        cmd = teacher.commands_for_skills()
        before = teacher.snapshot_state()
        a0 = teacher.act(cmd, advance=False)
        after = teacher.snapshot_state()
        self.assertTrue(torch.equal(before["blend"], after["blend"]))
        self.assertTrue(torch.equal(before["prev_params"], after["prev_params"]))
        self.assertTrue(torch.equal(before["jump_t"], after["jump_t"]))
        a1 = teacher.act(cmd, advance=True)
        self.assertTrue(torch.allclose(a0, a1))
        self.assertFalse(torch.equal(before["blend"], teacher._blend))

    def test_restore_only_masked_envs(self):
        teacher = TrotTeacher(num_envs=2, device=torch.device("cpu"))
        teacher.set_skill("forward")
        cmd = teacher.commands_for_skills()
        snap = teacher.snapshot_state()
        _ = teacher.act(cmd, advance=True)
        use_expert = torch.tensor([True, False])
        teacher.restore_state(snap, env_mask=~use_expert)
        self.assertFalse(torch.equal(snap["blend"][0], teacher._blend[0]))
        self.assertTrue(torch.equal(snap["blend"][1], teacher._blend[1]))

if __name__ == "__main__":
    unittest.main()
