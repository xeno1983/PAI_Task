"""Unit tests for GAIL discriminator helpers (no Genesis)."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.config import CPG_PARAM_DIM, NUM_ACTIONS, NUM_OBS
from pick_and_carry.curriculum import locomotion_action_mask
from pick_and_carry.gail import Discriminator, apply_action_mask, train_discriminator


class GailTests(unittest.TestCase):
    def test_reward_shape_and_positivity(self):
        disc = Discriminator(8, 4, hidden=(32, 32))
        obs = torch.randn(16, 8)
        act = torch.randn(16, 4)
        reward = disc.gail_reward(obs, act)
        self.assertEqual(tuple(reward.shape), (16,))
        self.assertTrue(torch.all(reward >= 0))

    def test_action_mask_zeros_unused(self):
        act = torch.ones(2, 4)
        mask = torch.tensor([True, True, False, False])
        masked = apply_action_mask(act, mask)
        self.assertTrue(torch.all(masked[:, :2] == 1))
        self.assertTrue(torch.all(masked[:, 2:] == 0))

    def test_locomotion_mask_keeps_cpg_only(self):
        act = torch.ones(3, NUM_ACTIONS)
        mask = locomotion_action_mask(NUM_ACTIONS)
        masked = apply_action_mask(act, mask)
        self.assertTrue(torch.all(masked[:, :CPG_PARAM_DIM] == 1))
        self.assertTrue(torch.all(masked[:, CPG_PARAM_DIM:] == 0))

    def test_discriminator_accepts_cpg_action_dim(self):
        disc = Discriminator(NUM_OBS, NUM_ACTIONS, hidden=(32, 32))
        obs = torch.randn(4, NUM_OBS)
        act = torch.randn(4, NUM_ACTIONS)
        reward = disc.gail_reward(obs, act)
        self.assertEqual(tuple(reward.shape), (4,))

    def test_discriminator_prefers_expert_after_train(self):
        torch.manual_seed(0)
        disc = Discriminator(4, 2, hidden=(64, 64))
        opt = torch.optim.Adam(disc.parameters(), lr=1e-2)
        expert_obs = torch.randn(256, 4)
        expert_act = expert_obs[:, :2]  # simple expert rule
        policy_obs = torch.randn(256, 4)
        policy_act = torch.randn(256, 2)
        with torch.no_grad():
            before = float(disc.gail_reward(expert_obs[:32], expert_act[:32]).mean())
        metrics = train_discriminator(
            disc,
            opt,
            expert_obs,
            expert_act,
            policy_obs,
            policy_act,
            epochs=20,
            batch_size=64,
        )
        with torch.no_grad():
            after_e = float(disc.gail_reward(expert_obs[:32], expert_act[:32]).mean())
            after_p = float(disc.gail_reward(policy_obs[:32], policy_act[:32]).mean())
        self.assertGreater(metrics["disc_acc"], 0.6)
        self.assertGreater(after_e, after_p)
        self.assertGreaterEqual(after_e, before - 1.0)


if __name__ == "__main__":
    unittest.main()
