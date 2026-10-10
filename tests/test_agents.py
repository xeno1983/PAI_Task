"""Agent helpers that do not need Genesis."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.agents import ActorCritic, ReplayBuffer, compute_gae
from pick_and_carry.config import NUM_ACTIONS, NUM_OBS


class AgentTests(unittest.TestCase):
    def test_policy_bounds_and_gae(self):
        agent = ActorCritic(NUM_OBS, NUM_ACTIONS, hidden=(32, 32))
        obs = torch.zeros(4, NUM_OBS)
        action, log_prob, value = agent.act(obs)
        self.assertEqual(tuple(action.shape), (4, NUM_ACTIONS))
        self.assertLess(float(action.detach().abs().max()), 1.0)
        self.assertEqual(tuple(log_prob.shape), (4,))
        self.assertEqual(tuple(value.shape), (4,))
        log_prob_2, entropy, value_2 = agent.evaluate(obs, action.detach())
        self.assertEqual(tuple(entropy.shape), (4,))
        self.assertEqual(tuple(value_2.shape), (4,))
        self.assertTrue(torch.isfinite(log_prob_2).all())

        rewards = torch.tensor([[1.0], [0.0]])
        values = torch.tensor([[0.5], [0.25]])
        boot = torch.tensor([[0.0], [0.0]])
        dones = torch.tensor([[1.0], [1.0]])
        advantages, returns = compute_gae(rewards, values, boot, dones, 0.99, 0.95)
        self.assertTrue(torch.allclose(advantages[0], rewards[0] - values[0]))
        self.assertTrue(torch.allclose(returns[0], rewards[0]))

    def test_replay_wrap(self):
        buffer = ReplayBuffer(4, 2, 1, torch.device("cpu"))
        obs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        actions = torch.zeros(3, 1)
        rewards = torch.arange(3, dtype=torch.float32)
        dones = torch.zeros(3)
        buffer.add(obs, actions, rewards, obs + 1, dones)
        self.assertEqual(buffer.size, 3)
        buffer.add(obs, actions, rewards, obs + 1, dones)
        self.assertEqual(buffer.size, 4)
        sample = buffer.sample(4)
        self.assertEqual(sample[0].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
