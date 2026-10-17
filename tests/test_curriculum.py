"""Curriculum rewards and layouts. No simulator."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.config import layout_dim
from pick_and_carry.curriculum import (
    BALL_RADIUS,
    CATCH_LEVELS,
    REACH_X,
    REACH_Y,
    REACH_Z,
    STAGES,
    ball_in_base,
    earlier_stages,
    feet_air_time_reward,
    holds_arm_pose,
    later_stages,
    locomotion_reward,
    manipulation_reward,
    retreat_choice,
    sample_ball,
    sample_command,
    sample_reach_target,
    stage_layouts,
    stage_log_dir,
    walk_score)


class LayoutTests(unittest.TestCase):
    def test_the_stages_are_walk_reach_the_box_and_catch(self):
        self.assertEqual(STAGES, ("walk", "reach", "haul", "carry", "catch"))
        self.assertEqual(CATCH_LEVELS, ("static", "slow", "random", "fly", "noisy"))

    def test_only_walk_leaves_the_arm_to_the_environment(self):
        self.assertTrue(holds_arm_pose("walk"))
        for stage in ("reach", "haul", "carry", "catch"):
            self.assertFalse(holds_arm_pose(stage))
        for gone in ("stand", "sit", "jump", "rough"):
            with self.assertRaises(ValueError):
                holds_arm_pose(gone)

    def test_every_stage_uses_the_arm_layout(self):
        from pick_and_carry.arms import layouts

        shared = layouts(6)
        for stage in STAGES:
            self.assertEqual(stage_layouts(stage, 6), shared)
        self.assertEqual(layout_dim(shared[0]), 81)
        self.assertEqual(layout_dim(shared[1]), 19)


class CommandTests(unittest.TestCase):
    def test_walk_commands_include_standing_still(self):
        torch.manual_seed(0)
        command = sample_command(4000, stand_share=0.2)
        still = (command.abs().sum(dim=-1) == 0).float().mean().item()
        self.assertAlmostEqual(still, 0.2, delta=0.03)
        self.assertEqual(float(sample_command(500).eq(0).all(dim=-1).float().sum()), 0.0)

    def test_reach_targets_stay_in_the_box_in_front_of_the_body(self):
        base = torch.tensor([[1.0, 2.0, 0.42]]).expand(500, 3)
        target = sample_reach_target(base)
        ahead = target[:, 0] - 1.0
        aside = target[:, 1] - 2.0
        self.assertGreaterEqual(float(ahead.min()), REACH_X[0] - 1e-6)
        self.assertLessEqual(float(ahead.max()), REACH_X[1] + 1e-6)
        self.assertGreaterEqual(float(aside.min()), REACH_Y[0] - 1e-6)
        self.assertLessEqual(float(aside.max()), REACH_Y[1] + 1e-6)
        self.assertGreaterEqual(float(target[:, 2].min()), REACH_Z[0] - 1e-6)
        self.assertLessEqual(float(target[:, 2].max()), REACH_Z[1] + 1e-6)

    def test_a_thrown_ball_passes_through_the_reach_box(self):
        torch.manual_seed(1)
        base = torch.tensor([[0.0, 0.0, 0.42]]).expand(300, 3)
        for level in ("fly", "noisy"):
            position, linear, angular = sample_ball(level, base)
            self.assertEqual(float(angular.abs().max()), 0.0)
            self.assertGreater(float(position[:, 0].min()), 1.3)
            self.assertGreater(float(position[:, 2].min()), 0.45)
            # Under gravity, find the point of the path nearest the front of the body.
            crossing = []
            for t in torch.linspace(0.6, 1.3, 71):
                point = position + linear * t
                point[:, 2] = point[:, 2] - 0.5 * 9.81 * t * t
                crossing.append(point)
            path = torch.stack(crossing)
            reach = (
                (path[..., 0] >= REACH_X[0] - 0.02)
                & (path[..., 0] <= REACH_X[1] + 0.02)
                & (path[..., 2] >= REACH_Z[0] - 0.05)
                & (path[..., 2] <= REACH_Z[1] + 0.05)
            ).any(dim=0)
            self.assertTrue(bool(reach.all()), level)
            # It comes toward the robot.
            self.assertLess(float(linear[:, 0].max()), 0.0)

    def test_the_ball_view_has_velocity_and_a_short_prediction(self):
        base_pos = torch.zeros(2, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, 4)
        ball = torch.tensor([[1.0, 0.0, 0.8], [1.0, 0.0, BALL_RADIUS]])
        velocity = torch.tensor([[-2.0, 0.0, 1.0], [-0.3, 0.0, 0.0]])
        seen, seen_vel, ahead = ball_in_base(ball, velocity, base_pos, quat, "fly")
        self.assertTrue(torch.allclose(seen, ball))
        self.assertTrue(torch.allclose(seen_vel, velocity))
        # Flying: 0.3 s along the path and a fall of g t^2 / 2. Rolling: a straight line on the ground.
        self.assertAlmostEqual(float(ahead[0, 0]), 1.0 - 0.6, places=4)
        self.assertAlmostEqual(float(ahead[0, 2]), 0.8 + 0.3 - 0.5 * 9.81 * 0.09, places=4)
        self.assertAlmostEqual(float(ahead[1, 0]), 1.0 - 0.09, places=4)
        self.assertAlmostEqual(float(ahead[1, 2]), BALL_RADIUS, places=4)

    def test_only_the_noisy_level_blurs_the_ball_view(self):
        base_pos = torch.zeros(64, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(64, 4)
        ball = torch.ones(64, 3)
        velocity = torch.ones(64, 3)
        clean, _, _ = ball_in_base(ball, velocity, base_pos, quat, "fly")
        blurred, _, _ = ball_in_base(ball, velocity, base_pos, quat, "noisy")
        self.assertEqual(float((clean - ball).abs().max()), 0.0)
        self.assertGreater(float((blurred - ball).abs().max()), 0.0)

    def test_slow_ball_rolls_toward_the_robot(self):
        base = torch.tensor([[0.0, 0.0, 0.4]])
        position, linear, _ = sample_ball("slow", base)
        self.assertGreater(position[0, 0].item(), 1.0)
        self.assertLess(linear[0, 0].item(), 0.0)

    def test_static_ball_does_not_move(self):
        base = torch.zeros(2, 3)
        _, linear, angular = sample_ball("static", base)
        self.assertEqual(float(linear.abs().max()), 0.0)
        self.assertEqual(float(angular.abs().max()), 0.0)


def _walk_reward(**overrides):
    command = torch.tensor([[0.0, 0.0, 0.0]])
    values = dict(
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.zeros(1, 3),
        command=command,
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        base_height=torch.tensor([0.30]),
        leg_actions=torch.zeros(1, 12),
        prev_leg_actions=torch.zeros(1, 12),
        leg_vel=torch.zeros(1, 12),
        prev_leg_vel=torch.zeros(1, 12),
        leg_torque=torch.zeros(1, 12),
        foot_contact=torch.ones(1, 4),
        air_time_reward=torch.zeros(1),
        collisions=torch.zeros(1),
        body_contact=torch.zeros(1, dtype=torch.bool),
        max_tilt_deg=60.0,
        min_base_height=0.18,
        dt=0.02)
    values.update(overrides)
    return locomotion_reward(**values)


class RewardTests(unittest.TestCase):
    def test_matching_the_command_scores_higher(self):
        command = sample_command(1)
        matched, _, _ = _walk_reward(
            base_lin_vel=command.new_tensor([[command[0, 0], command[0, 1], 0.0]]),
            base_ang_vel=command.new_tensor([[0.0, 0.0, command[0, 2]]]),
            command=command,
            foot_contact=torch.zeros(1, 4))
        missed, _, fallen = _walk_reward(
            command=command.new_tensor([[0.5, 0.0, 0.0]]),
            foot_contact=torch.zeros(1, 4))
        self.assertGreater(matched.item(), missed.item())
        self.assertFalse(bool(fallen))

    def test_a_diagonal_step_is_logged_but_a_shuffle_is_not(self):
        command = torch.tensor([[0.5, 0.0, 0.0]])
        _, trot_terms, _ = _walk_reward(command=command, foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
        _, shuffle_terms, _ = _walk_reward(command=command, foot_contact=torch.ones(1, 4))
        self.assertEqual(trot_terms["gait_score"].item(), 1.0)
        self.assertEqual(shuffle_terms["gait_score"].item(), 0.0)

    def test_a_long_step_earns_air_time_and_the_reward_follows(self):
        command = torch.tensor([[0.5, 0.0, 0.0]])
        stepping, _, _ = _walk_reward(command=command, air_time_reward=torch.tensor([0.3]))
        shuffling, _, _ = _walk_reward(command=command, air_time_reward=torch.tensor([-0.3]))
        self.assertGreater(stepping.item(), shuffling.item())

    def test_a_level_body_at_the_target_height_scores_higher_than_a_crouch(self):
        # Compare on a command the robot is meeting, so the positive part is not clipped away.
        command = torch.tensor([[0.5, 0.0, 0.0]])
        moving = dict(command=command, base_lin_vel=torch.tensor([[0.5, 0.0, 0.0]]))
        level, level_terms, _ = _walk_reward(**moving)
        low, low_terms, _ = _walk_reward(base_height=torch.tensor([0.22]), **moving)
        self.assertGreater(level.item(), low.item())
        self.assertGreater(level_terms["steady"].item(), low_terms["steady"].item())

    def test_a_steady_action_scores_higher_than_a_jump(self):
        moving = dict(command=torch.tensor([[0.5, 0.0, 0.0]]), base_lin_vel=torch.tensor([[0.5, 0.0, 0.0]]))
        held, held_terms, _ = _walk_reward(leg_actions=torch.ones(1, 12), prev_leg_actions=torch.ones(1, 12), **moving)
        jumped, jumped_terms, _ = _walk_reward(
            leg_actions=torch.ones(1, 12), prev_leg_actions=torch.full((1, 12), -1.0), **moving
        )
        self.assertGreater(held.item(), jumped.item())
        self.assertGreater(jumped_terms["jerk"].item(), held_terms["jerk"].item())

    def test_penalties_cannot_push_a_step_below_zero_before_the_fall_cost(self):
        reward, terms, fallen = _walk_reward(leg_torque=torch.full((1, 12), 20.0), collisions=torch.tensor([4.0]))
        self.assertFalse(bool(fallen))
        self.assertEqual(reward.item(), 0.0)
        self.assertEqual(terms["termination"].item(), 0.0)

    def test_touching_the_ground_with_the_body_ends_the_episode(self):
        reward, terms, fallen = _walk_reward(body_contact=torch.tensor([True]))
        self.assertTrue(bool(fallen))
        self.assertLess(reward.item(), 0.0)
        self.assertEqual(terms["termination"].item(), -5.0)

    def test_the_arm_never_enters_the_penalties(self):
        # Only leg actions are passed in, so arm noise cannot change the reward.
        self.assertNotIn("arm", " ".join(_walk_reward()[1]))


class WalkScoreTests(unittest.TestCase):
    def _score(self, command, lin, yaw=0.0):
        command = torch.tensor([command], dtype=torch.float32)
        lin_vel = torch.tensor([[lin[0], lin[1], 0.0]], dtype=torch.float32)
        ang_vel = torch.tensor([[0.0, 0.0, yaw]], dtype=torch.float32)
        return float(walk_score(command, lin_vel, ang_vel))

    def test_standing_still_scores_zero_on_a_moving_command(self):
        self.assertLess(self._score((0.4, 0.0, 0.0), (0.0, 0.0)), 0.01)
        self.assertLess(self._score((0.0, 0.0, 0.5), (0.0, 0.0), 0.0), 0.01)

    def test_the_commanded_speed_scores_one(self):
        self.assertAlmostEqual(self._score((0.4, 0.0, 0.0), (0.4, 0.0)), 1.0, places=4)
        self.assertAlmostEqual(self._score((0.0, 0.0, 0.5), (0.0, 0.0), 0.5), 1.0, places=4)

    def test_half_speed_scores_half_and_wrong_ways_score_zero(self):
        self.assertAlmostEqual(self._score((0.4, 0.0, 0.0), (0.2, 0.0)), 0.5, places=4)
        self.assertEqual(self._score((0.4, 0.0, 0.0), (-0.4, 0.0)), 0.0)
        self.assertEqual(self._score((0.4, 0.0, 0.0), (0.8, 0.0)), 0.0)

    def test_a_still_command_rewards_staying_still(self):
        self.assertGreater(self._score((0.0, 0.0, 0.0), (0.0, 0.0)), 0.99)
        self.assertLess(self._score((0.0, 0.0, 0.0), (0.3, 0.0)), 0.01)

    def test_both_parts_are_averaged(self):
        self.assertAlmostEqual(self._score((0.4, 0.0, 0.5), (0.4, 0.0), 0.0), 0.5, places=4)

    def test_standing_still_on_training_commands_is_far_below_the_difficulty_threshold(self):
        torch.manual_seed(0)
        command = sample_command(4000, stand_share=0.2)
        zero = torch.zeros(4000, 3)
        score = float(walk_score(command, zero, zero).mean())
        self.assertLess(score, 0.35)
        tracking = [float(_walk_reward(command=command[i : i + 1])[1]["tracking"]) for i in range(300)]
        self.assertGreater(sum(tracking) / len(tracking), 0.5)


class AirTimeTests(unittest.TestCase):
    def _step(self, air, last, contact, command, steps=1):
        reward = torch.zeros(1)
        for _ in range(steps):
            reward, air, last = feet_air_time_reward(air, last, contact, command, 0.02, 0.5)
        return reward, air, last

    def test_a_foot_in_the_air_collects_time_and_pays_on_touchdown(self):
        command = torch.tensor([[0.5, 0.0, 0.0]])
        air = torch.zeros(1, 4)
        last = torch.zeros(1, 4, dtype=torch.bool)
        up = torch.zeros(1, 4)
        reward, air, last = self._step(air, last, up, command, steps=40)
        self.assertEqual(reward.item(), 0.0)
        self.assertAlmostEqual(air[0, 0].item(), 0.8, places=4)
        down = torch.zeros(1, 4)
        down[0, 0] = 1.0
        reward, air, last = feet_air_time_reward(air, last, down, command, 0.02, 0.5)
        # 0.82 s in the air on the one foot that landed; the others stay in the air.
        self.assertAlmostEqual(reward.item(), 0.32, places=4)
        self.assertEqual(air[0, 0].item(), 0.0)

    def test_a_short_step_costs_the_shortfall(self):
        command = torch.tensor([[0.5, 0.0, 0.0]])
        air = torch.full((1, 4), 0.1)
        last = torch.zeros(1, 4, dtype=torch.bool)
        contact = torch.ones(1, 4)
        reward, _, _ = feet_air_time_reward(air, last, contact, command, 0.02, 0.5)
        self.assertLess(reward.item(), 0.0)

    def test_nothing_is_paid_without_a_speed_command(self):
        air = torch.full((1, 4), 0.1)
        last = torch.zeros(1, 4, dtype=torch.bool)
        reward, _, _ = feet_air_time_reward(air, last, torch.ones(1, 4), torch.zeros(1, 3), 0.02, 0.5)
        self.assertEqual(reward.item(), 0.0)

    def test_a_ball_in_the_hand_is_caught(self):
        hand = torch.tensor([[0.4, 0.0, 0.3]])
        _, terms, caught, _, _ = manipulation_reward(
            stage="catch",
            ee_pos=hand,
            object_pos=hand.clone(),
            object_vel=torch.zeros(1, 3),
            base_lin_vel_world=torch.zeros(1, 3),
            base_lin_vel=torch.zeros(1, 3),
            base_ang_vel=torch.zeros(1, 3),
            command=torch.zeros(1, 3),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            base_height=torch.tensor([0.4]),
            grasped=torch.tensor([False]),
            closed=torch.tensor([True]),
            actions=torch.zeros(1, 11),
            prev_distance=torch.tensor([0.0]),
            max_tilt_deg=60.0,
            min_base_height=0.18,
            dt=0.02)
        self.assertTrue(bool(caught))
        self.assertGreater(terms["catch"].item(), 0.0)

    def test_walk_observation_matches_the_layout(self):
        from pick_and_carry.observation import pack_observation

        observation, _ = stage_layouts("walk", 6)
        obs = pack_observation(
            leg_pos=torch.zeros(2, 12),
            leg_vel=torch.zeros(2, 12),
            projected_gravity=torch.zeros(2, 3),
            base_lin_vel=torch.zeros(2, 3),
            base_ang_vel=torch.zeros(2, 3),
            base_height=torch.zeros(2, 1),
            foot_contact=torch.zeros(2, 4),
            arm_pos=torch.zeros(2, 6),
            arm_vel=torch.zeros(2, 6),
            gripper_pos=torch.zeros(2, 1),
            gripper_vel=torch.zeros(2, 1),
            ee_pos=torch.zeros(2, 3),
            object_pos=torch.zeros(2, 3),
            goal_pos=torch.zeros(2, 3),
            grasped=torch.zeros(2, dtype=torch.bool),
            phase=torch.zeros(2, dtype=torch.long),
            command=torch.zeros(2, 3),
            last_leg_action=torch.zeros(2, 12))
        self.assertEqual(obs.shape, (2, layout_dim(observation)))


class TransferTests(unittest.TestCase):
    def test_each_stage_points_at_the_previous_skill(self):
        self.assertEqual(earlier_stages("walk"), [])
        self.assertEqual(earlier_stages("reach"), [("walk", "static")])
        self.assertEqual(earlier_stages("haul"), [("reach", "static"), ("walk", "static")])
        self.assertEqual(
            earlier_stages("carry"), [("haul", "static"), ("reach", "static"), ("walk", "static")]
        )
        # Catch grows from reach, not from the box.
        self.assertEqual(earlier_stages("catch", "static"), [("reach", "static"), ("walk", "static")])
        self.assertEqual(
            earlier_stages("catch", "fly"),
            [("catch", "random"), ("catch", "slow"), ("catch", "static"), ("reach", "static"), ("walk", "static")])
        self.assertEqual(later_stages("walk")[0], ("reach", "static"))
        self.assertEqual(later_stages("reach")[0], ("haul", "static"))
        self.assertEqual(later_stages("catch", "slow"), [("catch", "random"), ("catch", "fly"), ("catch", "noisy")])
        self.assertEqual(later_stages("catch", "noisy"), [])

    def test_reach_starts_from_walk_until_it_has_its_own_weights(self):
        import tempfile
        from pathlib import Path

        import torch

        from pick_and_carry.checkpoints import resolve_checkpoint

        def save(directory: Path, name: str, stage: str, obs: int = 81, act: int = 19, arm: str = "z1") -> None:
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"obs_dim": obs, "act_dim": act, "arm": arm, "curriculum": stage},
                directory / name)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            walk = stage_log_dir("walk", root=root)
            reach = stage_log_dir("reach", root=root)
            haul = stage_log_dir("haul", root=root)
            slow = stage_log_dir("catch", "slow", root=root)
            save(walk, "ppo_0001.pt", "walk", arm="panda")
            found = resolve_checkpoint(reach, "latest", 81, 19, "z1", "reach", root=root)
            self.assertIsNone(found)
            save(walk, "ppo_0002.pt", "walk")
            found = resolve_checkpoint(reach, "latest", 81, 19, "z1", "reach", root=root)
            self.assertEqual(found, walk / "ppo_0002.pt")
            save(reach, "ppo_0004.pt", "reach")
            found = resolve_checkpoint(reach, "latest", 81, 19, "z1", "reach", root=root)
            self.assertEqual(found, reach / "ppo_0004.pt")
            # The catch levels start from reach, and not from the box task.
            save(haul, "ppo_0001.pt", "haul")
            found = resolve_checkpoint(slow, "latest", 81, 19, "z1", "catch", "slow", root=root)
            self.assertEqual(found, reach / "ppo_0004.pt")
            # Practicing walk again prefers the harder stage's weights.
            found = resolve_checkpoint(walk, "latest", 81, 19, "z1", "walk", root=root, retreat=True)
            self.assertEqual(found, reach / "ppo_0004.pt")
            found = resolve_checkpoint(walk, "latest", 81, 19, "z1", "walk", root=root)
            self.assertEqual(found, walk / "ppo_0002.pt")

    def test_latest_checkpoint_is_numeric_not_lexicographic(self):
        """``ppo_9980.pt`` must not beat ``ppo_29980.pt`` under string sort."""
        import tempfile
        from pathlib import Path

        import torch

        from pick_and_carry.checkpoints import latest_matching

        def save(directory: Path, name: str, iteration: int) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "obs_dim": 81,
                    "act_dim": 19,
                    "arm": "z1",
                    "curriculum": "walk",
                    "iteration": iteration,
                    "model": {},
                },
                directory / name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            walk = Path(tmp) / "logs" / "ppo_curriculum_walk"
            save(walk, "ppo_9980.pt", 9980)
            save(walk, "ppo_10000.pt", 10000)
            save(walk, "ppo_29980.pt", 29980)
            found = latest_matching(walk, 81, 19, "z1", "walk")
            self.assertEqual(found, walk / "ppo_29980.pt")

    def test_a_removed_stage_has_no_folder(self):
        for gone in ("stand", "sit", "jump", "rough"):
            with self.assertRaises(ValueError):
                stage_log_dir(gone)

    def test_the_trainer_ranks_the_ladder(self):
        from pick_and_carry.checkpoints import stage_rank

        ranks = [stage_rank("walk", "static"), stage_rank("reach", "static"), stage_rank("haul", "static")]
        self.assertEqual(ranks, sorted(ranks))
        self.assertLess(ranks[-1], stage_rank("catch", "static"))
        self.assertLess(stage_rank("catch", "slow"), stage_rank("catch", "fly"))
        self.assertEqual(stage_rank("carry", "static"), 0)


class DifficultyTests(unittest.TestCase):
    def test_walk_starts_flat_and_climbs_with_the_score(self):
        from pick_and_carry.difficulty import Difficulty

        class Env:
            mix = None

            def set_terrain_mix(self, value):
                self.mix = value

        env = Env()
        difficulty = Difficulty("walk", "static")
        difficulty.start(env)
        self.assertEqual(env.mix, 0.0)
        for _ in range(8):
            difficulty.update(env, 0.9)
        self.assertEqual(env.mix, 0.5)
        for _ in range(8):
            difficulty.update(env, 0.9)
        self.assertEqual(env.mix, 1.0)

    def test_catch_climbs_back_toward_the_selected_ball(self):
        from pick_and_carry.difficulty import Difficulty

        class Env:
            level = None

            def set_catch_level(self, value):
                self.level = value

        env = Env()
        difficulty = Difficulty("catch", "fly")
        for _ in range(8):
            difficulty.update(env, 0.0)
        self.assertEqual(env.level, "random")


class TrainerSetupTests(unittest.TestCase):
    def test_minibatch_is_a_quarter_of_the_batch_unless_set(self):
        from types import SimpleNamespace

        from pick_and_carry.train_ppo import minibatch_size_for

        self.assertEqual(minibatch_size_for(SimpleNamespace(minibatch_size=0, num_mini_batches=4), 8192), 2048)
        self.assertEqual(minibatch_size_for(SimpleNamespace(minibatch_size=512, num_mini_batches=4), 8192), 512)

    def test_leg_noise_is_wide_and_arm_noise_is_narrow(self):
        from pick_and_carry.train_ppo import initial_action_std

        std = initial_action_std(19, 1.0, 0.4)
        self.assertEqual(std[:12].tolist(), [1.0] * 12)
        for value in std[12:].tolist():
            self.assertAlmostEqual(value, 0.4, places=6)

    def test_old_weights_are_moved_aside_but_matching_ones_stay(self):
        import tempfile
        from pathlib import Path

        from pick_and_carry.checkpoints import archive_incompatible_logs

        def save(directory: Path, name: str, obs: int) -> None:
            torch.save(
                {"obs_dim": obs, "act_dim": 19, "arm": "z1", "curriculum": "walk"},
                directory / name)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertIsNone(archive_incompatible_logs(folder, 81, 19, "z1", "walk"))
            save(folder, "ppo_0100.pt", 69)
            (folder / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
            (folder / "events.out.tfevents.1").write_text("x", encoding="utf-8")
            moved = archive_incompatible_logs(folder, 81, 19, "z1", "walk")
            self.assertIsNotNone(moved)
            self.assertTrue((moved / "ppo_0100.pt").is_file())
            self.assertTrue((moved / "metrics.jsonl").is_file())
            self.assertTrue((moved / "events.out.tfevents.1").is_file())
            self.assertEqual(list(folder.glob("ppo_*.pt")), [])
            save(folder, "ppo_0001.pt", 81)
            self.assertIsNone(archive_incompatible_logs(folder, 81, 19, "z1", "walk"))
            self.assertTrue((folder / "ppo_0001.pt").is_file())


class RetreatChoiceTests(unittest.TestCase):
    def test_low_scores_step_down_and_stop_at_the_easiest(self):
        current, poor, good = 2, 0, 0
        for score in (0.1, 0.1):
            current, poor, good = retreat_choice(
                score, current=current, target=2, poor_streak=poor, good_streak=good, patience=2
            )
        self.assertEqual(current, 1)
        current, poor, good = retreat_choice(0.1, current=0, target=2, poor_streak=1, good_streak=0, patience=2)
        current, poor, good = retreat_choice(0.1, current=current, target=2, poor_streak=poor, good_streak=good, patience=2)
        self.assertEqual(current, 0)

    def test_high_scores_climb_back_only_to_the_selected_task(self):
        current, poor, good = retreat_choice(0.9, current=0, target=2, poor_streak=0, good_streak=1, patience=2)
        self.assertEqual(current, 1)
        current, poor, good = retreat_choice(0.9, current=2, target=2, poor_streak=0, good_streak=5, patience=2)
        self.assertEqual(current, 2)

    def test_a_missing_score_does_not_move(self):
        self.assertEqual(retreat_choice(None, current=2, target=2, poor_streak=3, good_streak=1), (2, 3, 1))


class MissingPlaybackTests(unittest.TestCase):
    def test_catch_without_weights_names_the_reach_and_walk_folders(self):
        import tempfile
        from argparse import Namespace
        from pathlib import Path

        from pick_and_carry.config import EnvConfig
        from pick_and_carry.curriculum import stage_log_dir
        from pick_and_carry.checkpoints import missing_play_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = EnvConfig(curriculum="catch", catch_level="static", arm="z1")
            args = Namespace(
                play=True,
                resume="latest",
                retreat=False,
                log_dir=stage_log_dir("catch", "static", "z1", root=root))
            message = missing_play_checkpoint(args, cfg, root=root)
        self.assertIsNotNone(message)
        self.assertIn("再生できる重みがありません", message)
        self.assertIn("ppo_curriculum_reach", message)
        self.assertIn("ppo_curriculum_walk", message)
