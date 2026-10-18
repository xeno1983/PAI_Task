"""ValueDICE ν-network unit tests (no simulator)."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.config import NUM_ACTIONS, NUM_OBS
from pick_and_carry.valuedice import NuNet, apply_action_mask, soft_bound, train_nu


class ValueDiceTests(unittest.TestCase):
    def test_apply_action_mask_zeros_masked_dims(self):
        act = torch.ones(2, 4)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        out = apply_action_mask(act, mask)
        self.assertTrue(torch.allclose(out[:, :2], torch.ones(2, 2)))
        self.assertTrue(torch.allclose(out[:, 2:], torch.zeros(2, 2)))

    def test_nu_reward_shape_and_finite(self):
        nu = NuNet(8, 4, hidden=(32, 32))
        obs = torch.randn(5, 8)
        act = torch.randn(5, 4)
        reward = nu.valuedice_reward(obs, act)
        self.assertEqual(tuple(reward.shape), (5,))
        self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue(float(reward.detach().abs().max()) <= 5.0 + 1e-5)

    def test_soft_bound_keeps_gradients_at_extremes(self):
        x = torch.tensor([50.0, -50.0], requires_grad=True)
        y = soft_bound(x, 10.0)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(bool((x.grad.abs() > 0).all()))

    def test_train_nu_one_step_finite(self):
        nu = NuNet(NUM_OBS, NUM_ACTIONS, hidden=(32, 32))
        opt = torch.optim.Adam(nu.parameters(), lr=3e-4)
        expert_obs = torch.randn(64, NUM_OBS)
        expert_act = torch.randn(64, NUM_ACTIONS)
        policy_obs = torch.randn(64, NUM_OBS)
        policy_act = torch.randn(64, NUM_ACTIONS)
        metrics = train_nu(
            nu,
            opt,
            expert_obs,
            expert_act,
            policy_obs,
            policy_act,
            epochs=1,
            batch_size=32,
        )
        self.assertIn("nu_loss", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["nu_loss"])))

    def test_train_nu_next_state_residual_finite(self):
        nu = NuNet(8, 4, hidden=(32, 32))
        opt = torch.optim.Adam(nu.parameters(), lr=3e-4)
        n = 48
        metrics = train_nu(
            nu,
            opt,
            torch.randn(n, 8),
            torch.randn(n, 4),
            torch.randn(n, 8),
            torch.randn(n, 4),
            next_obs=torch.randn(n, 8),
            next_act=torch.randn(n, 4),
            gamma=0.99,
            epochs=1,
            batch_size=16,
        )
        self.assertIn("nu_delta", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["nu_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["nu_delta"])))

    def test_resume_helpers_accept_valuedice_latest(self):
        import tempfile
        from pathlib import Path

        from pick_and_carry.train_valuedice import _resolve_resume_path, _valuedice_iteration

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            out = log_dir / "valuedice_trot.pt"
            out.write_bytes(b"x")
            self.assertEqual(_resolve_resume_path("latest", log_dir, out), out)
            legacy = log_dir / "gail_trot.pt"
            missing = log_dir / "valuedice_trot.pt"
            missing.unlink()
            legacy.write_bytes(b"y")
            self.assertEqual(_resolve_resume_path("latest", log_dir, missing), legacy)
        self.assertEqual(_valuedice_iteration({"valuedice_iteration": 12, "iteration": 0}), 12)
        self.assertEqual(_valuedice_iteration({"gail_iteration": 7}), 7)


if __name__ == "__main__":
    unittest.main()
