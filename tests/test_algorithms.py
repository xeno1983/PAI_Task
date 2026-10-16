"""Unit tests for GRPO grouping and sagittal mirroring."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.algorithms import (
    ALGORITHMS,
    augment_symmetric,
    group_relative_advantages,
    mirror_action,
    mirror_observation,
    normalize_algo,
)
from pick_and_carry.curriculum import stage_log_dir


class AlgoNameTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_algo("symmetric-ppo"), "sym_ppo")
        self.assertEqual(tuple(ALGORITHMS), ("ppo", "grpo", "sym_ppo"))


class GroupRelativeTests(unittest.TestCase):
    def test_group_mean_is_zero(self):
        returns = torch.tensor([1.0, 3.0, 2.0, 4.0])
        adv = group_relative_advantages(returns, group_size=2)
        self.assertAlmostEqual(float(adv[0] + adv[1]), 0.0, places=5)
        self.assertAlmostEqual(float(adv[2] + adv[3]), 0.0, places=5)


class MirrorTests(unittest.TestCase):
    def test_mirror_twice_is_identity(self):
        obs = torch.randn(4, 81)
        action = torch.randn(4, 19)
        self.assertTrue(torch.allclose(mirror_observation(mirror_observation(obs)), obs, atol=1e-6))
        self.assertTrue(torch.allclose(mirror_action(mirror_action(action)), action, atol=1e-6))

    def test_leg_swap_and_hip_sign(self):
        obs = torch.zeros(1, 81)
        obs[0, 0] = 0.5  # FL hip
        obs[0, 3] = -0.2  # FR hip
        mirrored = mirror_observation(obs)
        self.assertAlmostEqual(float(mirrored[0, 0]), 0.2, places=5)
        self.assertAlmostEqual(float(mirrored[0, 3]), -0.5, places=5)

    def test_augment_doubles_batch(self):
        obs = torch.randn(3, 81)
        actions = torch.randn(3, 19)
        logp = torch.randn(3)
        returns = torch.randn(3)
        advantages = torch.randn(3)
        values = torch.randn(3)
        out = augment_symmetric(obs, actions, logp, returns, advantages, values)
        self.assertEqual(tuple(out[0].shape), (6, 81))
        self.assertEqual(tuple(out[1].shape), (6, 19))
        self.assertEqual(tuple(out[5].shape), (6,))


class StageLogDirAlgoTests(unittest.TestCase):
    def test_algo_changes_folder(self):
        self.assertEqual(stage_log_dir("walk", algo="ppo").as_posix(), "logs/ppo_curriculum_walk")
        self.assertEqual(stage_log_dir("walk", algo="grpo").as_posix(), "logs/grpo_curriculum_walk")
        self.assertEqual(stage_log_dir("carry", algo="sym_ppo").as_posix(), "logs/sym_ppo_z1")


if __name__ == "__main__":
    unittest.main()
