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
        self.assertEqual(tuple(ALGORITHMS), ("ppo", "grpo", "sym_ppo"))


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

    def test_monte_carlo_stops_at_done(self):
        rewards = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        dones = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        returns = monte_carlo_returns(rewards, dones, gamma=1.0)
        # Env 0 ends after step 1: returns are 2, 1, 1
        self.assertEqual([float(x) for x in returns[:, 0]], [2.0, 1.0, 1.0])
        # Env 1 never ends in the window: 3, 2, 1
        self.assertEqual([float(x) for x in returns[:, 1]], [3.0, 2.0, 1.0])


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
        from pick_and_carry.config import NUM_ACTIONS

        agent = ActorCritic(81, NUM_ACTIONS, hidden=(32, 32))
        obs = torch.randn(3, 81)
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


class StageLogDirAlgoTests(unittest.TestCase):
    def test_algo_changes_folder(self):
        self.assertEqual(stage_log_dir("walk", algo="ppo").as_posix(), "logs/ppo_curriculum_walk")
        self.assertEqual(stage_log_dir("walk", algo="grpo").as_posix(), "logs/grpo_curriculum_walk")
        self.assertEqual(stage_log_dir("carry", algo="sym_ppo").as_posix(), "logs/sym_ppo_z1")


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
