"""Unit tests for GRPO grouping and sagittal mirroring."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.agents import ActorCritic
from pick_and_carry.algorithms import (
    ALGORITHMS,
    augment_symmetric,
    group_relative_advantages,
    mirror_action,
    mirror_observation,
    monte_carlo_returns,
    normalize_algo,
)
from pick_and_carry.curriculum import stage_log_dir


class AlgoNameTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_algo("symmetric-ppo"), "sym_ppo")
        self.assertEqual(normalize_algo("dr-grpo"), "dr_grpo")
        self.assertEqual(tuple(ALGORITHMS), ("ppo", "grpo", "dr_grpo", "sym_ppo"))


class ActionMaskTests(unittest.TestCase):
    def test_masked_entropy_ignores_arm_dims(self):
        from pick_and_carry.config import CPG_PARAM_DIM, NUM_ACTIONS
        from pick_and_carry.curriculum import locomotion_action_mask

        agent = ActorCritic(81, NUM_ACTIONS, init_std=0.5)
        agent.set_action_mask(locomotion_action_mask(NUM_ACTIONS))
        with torch.no_grad():
            agent.actor.log_std.fill_(0.0)  # std = 1 for all dims
        obs = torch.zeros(4, 81)
        action = torch.zeros(4, NUM_ACTIONS)
        _, entropy_full = agent.actor.log_prob(obs, action, None)
        _, entropy_cpg = agent.actor.log_prob(obs, action, agent.action_mask)
        self.assertGreater(float(entropy_full.mean()), float(entropy_cpg.mean()))
        self.assertAlmostEqual(
            float(entropy_cpg.mean()),
            float(entropy_full.mean()) * CPG_PARAM_DIM / NUM_ACTIONS,
            places=3,
        )

    def test_easy_walk_log_dir(self):
        self.assertTrue(str(stage_log_dir("walk", easy=True)).endswith("curriculum_walk_easy"))
        self.assertTrue(str(stage_log_dir("walk", easy=False)).endswith("curriculum_walk"))

    def test_imitation_log_dir_is_sibling_of_walk(self):
        from pathlib import Path

        from pick_and_carry.curriculum import imitation_log_dir, nest_imitation_log_dir

        self.assertEqual(str(imitation_log_dir()).replace("\\", "/"), "logs/imitation")
        rl = stage_log_dir("walk", easy=True)
        self.assertEqual(nest_imitation_log_dir(rl), rl.parent / "imitation")
        self.assertEqual(rl.parent / "imitation", Path("logs") / "imitation")


class GroupRelativeTests(unittest.TestCase):
    def test_group_mean_is_zero(self):
        returns = torch.tensor([1.0, 3.0, 2.0, 4.0])
        adv = group_relative_advantages(returns, group_size=2)
        self.assertAlmostEqual(float(adv[0] + adv[1]), 0.0, places=5)
        self.assertAlmostEqual(float(adv[2] + adv[3]), 0.0, places=5)

    def test_dr_grpo_mean_scale_no_std(self):
        from pick_and_carry.algorithms import grpo_advantage_scale, is_critic_free_algo

        self.assertTrue(is_critic_free_algo("dr_grpo"))
        self.assertEqual(grpo_advantage_scale("dr_grpo"), "mean")
        self.assertEqual(grpo_advantage_scale("grpo"), "std")
        # Same mean offset, different spread: std-scale equalizes; mean-scale keeps magnitude.
        tight = torch.tensor([0.0, 2.0])
        wide = torch.tensor([0.0, 20.0])
        a_std = group_relative_advantages(torch.cat([tight, wide]), group_size=2, scale="std")
        a_mean = group_relative_advantages(torch.cat([tight, wide]), group_size=2, scale="mean")
        self.assertAlmostEqual(float(a_std[1]), -float(a_std[0]), places=5)
        self.assertAlmostEqual(float(a_mean[1]), 1.0, places=5)
        self.assertAlmostEqual(float(a_mean[3]), 10.0, places=5)
        self.assertGreater(abs(float(a_mean[3])), abs(float(a_std[3])))

    def test_incomplete_tail_uses_batch_stats(self):
        # group_size=3 → one leftover; must not zero itself via leftover-only mean.
        returns = torch.tensor([1.0, 2.0, 3.0, 10.0])
        adv = group_relative_advantages(returns, group_size=3, scale="mean")
        batch_mean = float(returns.mean())
        self.assertAlmostEqual(float(adv[3]), 10.0 - batch_mean, places=5)
        self.assertNotAlmostEqual(float(adv[3]), 0.0, places=5)

    def test_monte_carlo_stops_at_done(self):
        rewards = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        dones = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        returns = monte_carlo_returns(rewards, dones, gamma=1.0)
        # Env 0 ends after step 1: returns are 2, 1, 1
        self.assertEqual([float(x) for x in returns[:, 0]], [2.0, 1.0, 1.0])
        # Env 1 never ends in the window: 3, 2, 1
        self.assertEqual([float(x) for x in returns[:, 1]], [3.0, 2.0, 1.0])

    def test_stabilize_clips_extreme_advantages(self):
        from pick_and_carry.algorithms import stabilize_grpo_advantages, winsorize_returns

        # One catastrophic return among otherwise similar ones.
        returns = torch.tensor([-50.0, -0.02, -0.01, 0.0, 0.01, 0.02, -0.015, -0.005])
        soft = winsorize_returns(returns, quantile=0.25)
        self.assertGreater(float(soft.min()), -50.0)
        adv = stabilize_grpo_advantages(returns, group_size=4, return_quantile=0.25, adv_clip=2.0)
        self.assertLessEqual(float(adv.max()), 2.0 + 1e-5)
        self.assertGreaterEqual(float(adv.min()), -2.0 - 1e-5)


class MirrorTests(unittest.TestCase):
    def test_mirror_twice_is_identity(self):
        from pick_and_carry.config import NUM_ACTIONS

        obs = torch.randn(4, 81)
        action = torch.randn(4, NUM_ACTIONS)
        self.assertTrue(torch.allclose(mirror_observation(mirror_observation(obs)), obs, atol=1e-6))
        self.assertTrue(torch.allclose(mirror_action(mirror_action(action)), action, atol=1e-6))

    def test_leg_swap_and_hip_sign(self):
        obs = torch.zeros(1, 81)
        obs[0, 0] = 0.5  # FL hip
        obs[0, 3] = -0.2  # FR hip
        mirrored = mirror_observation(obs)
        self.assertAlmostEqual(float(mirrored[0, 0]), 0.2, places=5)
        self.assertAlmostEqual(float(mirrored[0, 3]), -0.5, places=5)

    def test_arm_yaw_and_wrist_axes_flip(self):
        from pick_and_carry.config import NUM_ACTIONS

        obs = torch.zeros(1, 81)
        action = torch.zeros(1, NUM_ACTIONS)
        obs[0, 38] = 0.3  # joint1
        obs[0, 42] = 0.4  # joint5
        obs[0, 43] = -0.5  # joint6
        # Arm block starts after CPG_PARAM_DIM (15): joints at 15, 19, 20.
        from pick_and_carry.config import CPG_PARAM_DIM

        j1, j5, j6 = CPG_PARAM_DIM, CPG_PARAM_DIM + 4, CPG_PARAM_DIM + 5
        action[0, j1] = 0.3
        action[0, j5] = 0.4
        action[0, j6] = -0.5
        mirrored_obs = mirror_observation(obs)
        mirrored_act = mirror_action(action)
        self.assertAlmostEqual(float(mirrored_obs[0, 38]), -0.3, places=5)
        self.assertAlmostEqual(float(mirrored_obs[0, 42]), -0.4, places=5)
        self.assertAlmostEqual(float(mirrored_obs[0, 43]), 0.5, places=5)
        self.assertAlmostEqual(float(mirrored_act[0, j1]), -0.3, places=5)
        self.assertAlmostEqual(float(mirrored_act[0, j5]), -0.4, places=5)
        self.assertAlmostEqual(float(mirrored_act[0, j6]), 0.5, places=5)

    def test_catch_command_keeps_z(self):
        obs = torch.zeros(1, 81)
        obs[0, 67] = 0.2  # predicted y
        obs[0, 68] = 0.5  # predicted z
        mirrored = mirror_observation(obs, command_has_yaw=False)
        self.assertAlmostEqual(float(mirrored[0, 67]), -0.2, places=5)
        self.assertAlmostEqual(float(mirrored[0, 68]), 0.5, places=5)
        walk = mirror_observation(obs, command_has_yaw=True)
        self.assertAlmostEqual(float(walk[0, 68]), -0.5, places=5)

    def test_augment_recomputes_mirrored_log_prob(self):
        from pick_and_carry.config import NUM_ACTIONS, OBS_LAYOUT, layout_slices

        agent = ActorCritic(81, NUM_ACTIONS, hidden=(32, 32))
        obs = torch.randn(3, 81)
        # All three ask to move so the batch still doubles.
        cmd = layout_slices(OBS_LAYOUT)["command"]
        obs[:, cmd] = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.5]])
        actions = torch.randn(3, NUM_ACTIONS)
        with torch.no_grad():
            old_logp, _ = agent.actor.log_prob(obs, actions)
        returns = torch.randn(3)
        advantages = torch.randn(3)
        values = torch.randn(3)
        out = augment_symmetric(
            obs, actions, old_logp, returns, advantages, values, agent=agent, command_has_yaw=True
        )
        self.assertEqual(tuple(out[0].shape), (6, 81))
        with torch.no_grad():
            expected, _ = agent.actor.log_prob(out[0][3:], out[1][3:])
        self.assertTrue(torch.allclose(out[2][3:], expected, atol=1e-5))
        self.assertFalse(torch.allclose(out[2][3:], old_logp, atol=1e-3))

    def test_augment_mirrors_only_moving_commands(self):
        from pick_and_carry.algorithms import moving_command_mask
        from pick_and_carry.config import NUM_ACTIONS, OBS_LAYOUT, layout_slices

        obs = torch.zeros(4, 81)
        cmd = layout_slices(OBS_LAYOUT)["command"]
        obs[0, cmd] = torch.tensor([0.5, 0.0, 0.0])  # move
        obs[1, cmd] = torch.tensor([0.0, 0.0, 0.0])  # still
        obs[2, cmd] = torch.tensor([0.0, 0.0, 0.5])  # yaw turn
        obs[3, cmd] = torch.tensor([0.05, 0.0, 0.0])  # below MOVE_LIN
        mask = moving_command_mask(obs, command_has_yaw=True)
        self.assertEqual([bool(x) for x in mask], [True, False, True, False])

        actions = torch.randn(4, NUM_ACTIONS)
        old_logp = torch.randn(4)
        returns = torch.arange(4, dtype=torch.float32)
        advantages = torch.arange(4, dtype=torch.float32) * 0.1
        values = torch.arange(4, dtype=torch.float32) * 0.01
        out = augment_symmetric(
            obs, actions, old_logp, returns, advantages, values, agent=None, command_has_yaw=True
        )
        # Original 4 + mirrors of indices 0 and 2 → 6.
        self.assertEqual(tuple(out[0].shape), (6, 81))
        self.assertTrue(torch.allclose(out[3][:4], returns))
        self.assertTrue(torch.allclose(out[3][4:], returns[mask]))

        still_only = obs[1:2]
        out_still = augment_symmetric(
            still_only,
            actions[1:2],
            old_logp[1:2],
            returns[1:2],
            advantages[1:2],
            values[1:2],
            agent=None,
            command_has_yaw=True,
        )
        self.assertEqual(tuple(out_still[0].shape), (1, 81))


class StageLogDirAlgoTests(unittest.TestCase):
    def test_algo_changes_folder(self):
        self.assertEqual(stage_log_dir("walk", algo="ppo").as_posix(), "logs/ppo_curriculum_walk")
        self.assertEqual(stage_log_dir("walk", algo="grpo").as_posix(), "logs/grpo_curriculum_walk")
        self.assertEqual(stage_log_dir("walk", algo="dr_grpo").as_posix(), "logs/dr_grpo_curriculum_walk")
        self.assertEqual(stage_log_dir("carry", algo="sym_ppo").as_posix(), "logs/sym_ppo_z1")


class CheckpointKindTests(unittest.TestCase):
    def test_rl_vs_imitation_blob(self):
        from pick_and_carry.checkpoints import is_imitation_checkpoint, is_rl_training_checkpoint

        bc = {"model": {}, "bc_trot": True, "iteration": 0}
        self.assertTrue(is_imitation_checkpoint(bc))
        self.assertFalse(is_rl_training_checkpoint(bc))
        ppo = {"model": {}, "optimizer": {}, "iteration": 1000, "algo": "ppo"}
        self.assertFalse(is_imitation_checkpoint(ppo))
        self.assertTrue(is_rl_training_checkpoint(ppo))
        legacy = {"model": {}, "iteration": 500}
        self.assertTrue(is_rl_training_checkpoint(legacy))


class PpoInitFromTests(unittest.TestCase):
    def test_resolve_init_from_searches_imitation_and_easy_sibling(self):
        import tempfile
        from pathlib import Path

        from pick_and_carry.train_ppo import resolve_init_from

        root = Path(tempfile.mkdtemp())
        logs = root / "logs"
        hard = logs / "ppo_curriculum_walk"
        easy = logs / "ppo_curriculum_walk_easy"
        imit = logs / "imitation"
        hard.mkdir(parents=True)
        easy.mkdir()
        imit.mkdir()
        bc = imit / "bc_trot.pt"
        bc.write_bytes(b"stub")
        # Chdir so imitation_log_dir() resolves relative logs/imitation under root.
        import os

        prev = os.getcwd()
        try:
            os.chdir(root)
            self.assertEqual(resolve_init_from("bc", hard).resolve(), bc.resolve())
            self.assertEqual(resolve_init_from("latest", hard).resolve(), bc.resolve())
            self.assertEqual(Path(resolve_init_from(str(bc), hard)).resolve(), bc.resolve())
        finally:
            os.chdir(prev)

    def test_bc_imitation_reward_penalizes_drift_from_teacher(self):
        from pick_and_carry.config import NUM_ACTIONS
        from pick_and_carry.train_ppo import bc_imitation_reward, freeze_actor_as_bc_teacher, resolve_bc_imit_coef

        agent = ActorCritic(81, NUM_ACTIONS, hidden=(32, 32))
        teacher = freeze_actor_as_bc_teacher(agent.actor)
        obs = torch.zeros(4, 81)
        match = teacher.mean_action(obs)
        drifted = match + 1.0
        close = bc_imitation_reward(teacher, obs, match, None, 0.1)
        far = bc_imitation_reward(teacher, obs, drifted, None, 0.1)
        self.assertTrue(torch.allclose(close, torch.zeros_like(close), atol=1e-5))
        self.assertTrue((far < close).all())
        args = type("A", (), {"bc_imit_coef": None})()
        self.assertEqual(resolve_bc_imit_coef(args, warmed_from_imitation=True), 0.1)
        self.assertEqual(resolve_bc_imit_coef(args, warmed_from_imitation=False), 0.0)
        args.bc_imit_coef = 0.0
        self.assertEqual(resolve_bc_imit_coef(args, warmed_from_imitation=True), 0.0)


if __name__ == "__main__":
    unittest.main()
