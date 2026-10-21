"""Agent helpers that do not need Genesis."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.agents import (
    ActorCritic,
    _ppo_policy_loss,
    adapt_learning_rate,
    compute_gae,
    gaussian_kl,
    ppo_update,
)
from pick_and_carry.config import NUM_ACTIONS, NUM_OBS


class AgentTests(unittest.TestCase):
    def test_policy_shapes_and_gae(self):
        agent = ActorCritic(NUM_OBS, NUM_ACTIONS, hidden=(32, 32))
        obs = torch.zeros(4, NUM_OBS)
        action, log_prob, value = agent.act(obs)
        self.assertEqual(tuple(action.shape), (4, NUM_ACTIONS))
        self.assertTrue(torch.isfinite(action).all())
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

    def test_actions_are_not_squashed_and_the_noise_starts_per_action(self):
        from pick_and_carry.config import CPG_PARAM_DIM

        arm_dims = NUM_ACTIONS - CPG_PARAM_DIM
        init = torch.cat((torch.full((CPG_PARAM_DIM,), 1.0), torch.full((arm_dims,), 0.4)))
        agent = ActorCritic(NUM_OBS, NUM_ACTIONS, hidden=(32, 32), init_std=init)
        std = agent.actor.std()
        self.assertTrue(torch.allclose(std[:CPG_PARAM_DIM], torch.ones(CPG_PARAM_DIM)))
        self.assertTrue(torch.allclose(std[CPG_PARAM_DIM:], torch.full((arm_dims,), 0.4)))
        action, _, _ = agent.act(torch.zeros(4096, NUM_OBS))
        # A tanh policy could never exceed 1; a Gaussian one does.
        self.assertGreater(float(action.abs().max()), 1.5)

    def test_the_noise_cannot_grow_past_one(self):
        agent = ActorCritic(NUM_OBS, NUM_ACTIONS, hidden=(32, 32))
        with torch.no_grad():
            agent.actor.log_std.fill_(3.0)
        self.assertLessEqual(float(agent.actor.std().max()), 1.0 + 1e-6)

    def test_the_mean_action_starts_near_the_reference_pose(self):
        agent = ActorCritic(NUM_OBS, NUM_ACTIONS, hidden=(32, 32))
        with torch.no_grad():
            mean = agent.act_mean(torch.randn(8, NUM_OBS))
        self.assertLess(float(mean.abs().max()), 0.2)



class PpoUpdateTests(unittest.TestCase):
    def test_identical_distributions_have_no_kl(self):
        mean = torch.randn(5, 3)
        std = torch.full((5, 3), 0.7)
        self.assertAlmostEqual(float(gaussian_kl(mean, std, mean, std)), 0.0, places=6)
        self.assertGreater(float(gaussian_kl(mean, std, mean + 1.0, std)), 0.0)

    def test_learning_rate_follows_the_kl_target(self):
        self.assertLess(adapt_learning_rate(1e-3, 0.05, 0.01), 1e-3)
        self.assertGreater(adapt_learning_rate(1e-3, 0.001, 0.01), 1e-3)
        self.assertEqual(adapt_learning_rate(1e-3, 0.01, 0.01), 1e-3)
        self.assertEqual(adapt_learning_rate(1e-5, 1.0, 0.01), 1e-5)
        self.assertEqual(adapt_learning_rate(1e-2, 1e-6, 0.01), 1e-2)

    def test_an_update_runs_with_the_clipped_value_loss_and_adaptive_rate(self):
        torch.manual_seed(0)
        agent = ActorCritic(NUM_OBS, NUM_ACTIONS, hidden=(32, 32))
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        obs = torch.randn(64, NUM_OBS)
        with torch.no_grad():
            action, log_prob, value = agent.act(obs)
        metrics = ppo_update(
            agent,
            optimizer,
            obs,
            action,
            log_prob,
            value + torch.randn(64),
            torch.randn(64),
            clip=0.2,
            epochs=2,
            minibatch_size=16,
            entropy_coef=0.01,
            value_coef=1.0,
            max_grad_norm=1.0,
            old_values=value,
            desired_kl=0.01,
        )
        for key in ("policy_loss", "value_loss", "entropy", "kl", "lr"):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])), key)
        self.assertGreaterEqual(metrics["kl"], 0.0)
        self.assertGreaterEqual(metrics["lr"], 1e-5)
        self.assertLessEqual(metrics["lr"], 1e-2)

    def test_the_rate_stays_put_without_a_kl_target(self):
        agent = ActorCritic(NUM_OBS, NUM_ACTIONS, hidden=(16,))
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        obs = torch.randn(32, NUM_OBS)
        with torch.no_grad():
            action, log_prob, value = agent.act(obs)
        metrics = ppo_update(
            agent,
            optimizer,
            obs,
            action,
            log_prob,
            value,
            torch.randn(32),
            clip=0.2,
            epochs=1,
            minibatch_size=8,
            entropy_coef=0.0,
            value_coef=1.0,
            max_grad_norm=1.0,
        )
        self.assertEqual(metrics["lr"], 3e-4)

    def test_dual_clip_bounds_negative_advantage_surrogate(self):
        # r ≫ 1 and A < 0: plain PPO would use r·A (huge); dual-clip floors at c·A.
        ratio = torch.tensor([100.0, 100.0])
        adv = torch.tensor([-2.0, 2.0])
        loss = float(_ppo_policy_loss(ratio, adv, clip=0.2))
        self.assertLess(abs(loss), 20.0)
        self.assertTrue(loss == loss)  # finite

    def test_extreme_old_log_prob_does_not_nan_the_actor(self):
        """Regression: unclamped ratio + A<0 made policy_loss explode to NaN (Sym PPO)."""
        torch.manual_seed(1)
        agent = ActorCritic(NUM_OBS, NUM_ACTIONS, hidden=(32, 32))
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        obs = torch.randn(64, NUM_OBS)
        with torch.no_grad():
            action, log_prob, value = agent.act(obs)
            # Pretend the behavior policy had wildly different log-probs.
            bad_old = log_prob - 80.0
            # Mix of positive and negative advantages (negative is the dangerous case).
            adv = torch.linspace(-3.0, 3.0, 64)
        metrics = ppo_update(
            agent,
            optimizer,
            obs,
            action,
            bad_old,
            value,
            adv,
            clip=0.2,
            epochs=3,
            minibatch_size=16,
            entropy_coef=0.0,
            value_coef=1.0,
            max_grad_norm=1.0,
            desired_kl=0.01,
            normalize_advantages=False,
        )
        with torch.no_grad():
            mean = agent.actor.mean_action(obs)
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(torch.tensor(metrics["policy_loss"])))
        self.assertLess(abs(metrics["policy_loss"]), 100.0)


if __name__ == "__main__":
    unittest.main()
