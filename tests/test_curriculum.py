"""Curriculum rewards and layouts. No simulator."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.config import layout_dim
from pick_and_carry.curriculum import (
    SIT_HEIGHT,
    STAND_HEIGHT,
    crouch_legs,
    earlier_stages,
    feet_air_time_reward,
    later_stages,
    locomotion_reward,
    manipulation_reward,
    pose_reward,
    posture_achieved,
    holds_arm_pose,
    retreat_choice,
    sample_ball,
    sample_command,
    stage_layouts,
    stage_log_dir,
)


class LayoutTests(unittest.TestCase):
    def test_leg_tasks_hold_the_reset_arm_pose(self):
        for stage in ("stand", "sit", "walk", "jump", "rough"):
            self.assertTrue(holds_arm_pose(stage))
        for stage in ("haul", "carry", "catch"):
            self.assertFalse(holds_arm_pose(stage))

    def test_every_stage_uses_the_arm_layout(self):
        from pick_and_carry.arms import layouts

        shared = layouts(6)
        for stage in ("carry", "stand", "sit", "walk", "jump", "rough", "haul", "catch"):
            self.assertEqual(stage_layouts(stage, 6), shared)
        self.assertEqual(layout_dim(shared[0]), 81)
        self.assertEqual(layout_dim(shared[1]), 19)


class CommandTests(unittest.TestCase):
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


class PostureTests(unittest.TestCase):
    def _reward(self, task: str, height: float, vz: float = 0.0):
        reward, terms, fallen = pose_reward(
            task,
            base_height=torch.tensor([height]),
            base_lin_vel=torch.zeros(1, 3),
            base_lin_vel_world=torch.tensor([[0.0, 0.0, vz]]),
            base_ang_vel=torch.zeros(1, 3),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            foot_contact=torch.ones(1, 4),
            actions=torch.zeros(1, 12),
            max_tilt_deg=60.0,
            min_base_height=0.18,
            dt=0.02,
        )
        return reward, terms, fallen

    def test_the_crouch_stays_inside_the_standing_action(self):
        from pick_and_carry.curriculum import CROUCH_FOLD

        standing = torch.tensor([0.0, 0.8, -1.5] * 4)
        folded = crouch_legs(standing)
        self.assertAlmostEqual(float(folded[1] - standing[1]), CROUCH_FOLD)
        self.assertAlmostEqual(float(standing[2] - folded[2]), CROUCH_FOLD)
        self.assertEqual(float(folded[0]), 0.0)
        self.assertLessEqual(float((folded - standing).abs().max()), 0.50)

    def test_standing_at_the_target_height_scores_higher_than_crouching(self):
        _, standing, _ = self._reward("stand", STAND_HEIGHT)
        _, low, _ = self._reward("stand", SIT_HEIGHT)
        self.assertGreater(standing["tracking"].item(), low["tracking"].item())

    def test_sitting_at_the_low_height_scores_higher_than_standing_tall(self):
        _, sitting, _ = self._reward("sit", SIT_HEIGHT)
        _, tall, _ = self._reward("sit", STAND_HEIGHT)
        self.assertGreater(sitting["tracking"].item(), tall["tracking"].item())

    def test_a_jump_lands_only_after_the_feet_leave_the_ground(self):
        height = torch.tensor([STAND_HEIGHT])
        still = torch.zeros(1, 3)
        down = torch.ones(1, 4)
        up = torch.zeros(1, 4)
        fallen = torch.tensor([False])
        landed, airborne = posture_achieved(
            "jump",
            base_height=height,
            base_lin_vel=still,
            base_ang_vel=still,
            foot_contact=down,
            fallen=fallen,
            was_airborne=torch.tensor([False]),
        )
        self.assertFalse(bool(landed))
        self.assertFalse(bool(airborne))
        _, airborne = posture_achieved(
            "jump",
            base_height=torch.tensor([STAND_HEIGHT + 0.1]),
            base_lin_vel=still,
            base_ang_vel=still,
            foot_contact=up,
            fallen=fallen,
            was_airborne=torch.tensor([False]),
        )
        self.assertTrue(bool(airborne))
        landed, _ = posture_achieved(
            "jump",
            base_height=height,
            base_lin_vel=still,
            base_ang_vel=still,
            foot_contact=down,
            fallen=fallen,
            was_airborne=torch.tensor([True]),
        )
        self.assertTrue(bool(landed))


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
        dt=0.02,
    )
    values.update(overrides)
    return locomotion_reward(**values)


class RewardTests(unittest.TestCase):
    def test_matching_the_command_scores_higher(self):
        command = sample_command(1)
        matched, _, _ = _walk_reward(
            base_lin_vel=command.new_tensor([[command[0, 0], command[0, 1], 0.0]]),
            base_ang_vel=command.new_tensor([[0.0, 0.0, command[0, 2]]]),
            command=command,
            foot_contact=torch.zeros(1, 4),
        )
        missed, _, fallen = _walk_reward(
            command=command.new_tensor([[0.5, 0.0, 0.0]]),
            foot_contact=torch.zeros(1, 4),
        )
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
            dt=0.02,
        )
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
            last_leg_action=torch.zeros(2, 12),
        )
        self.assertEqual(obs.shape, (2, layout_dim(observation)))


class TransferTests(unittest.TestCase):
    def test_each_stage_points_at_the_previous_skill(self):
        self.assertEqual(earlier_stages("stand"), [])
        self.assertEqual(earlier_stages("sit"), [("stand", "static")])
        self.assertEqual(earlier_stages("walk"), [("sit", "static"), ("stand", "static")])
        self.assertEqual(earlier_stages("carry"), [("haul", "static"), ("rough", "static"), ("jump", "static"), ("walk", "static"), ("sit", "static"), ("stand", "static")])
        self.assertEqual(earlier_stages("catch", "static")[0], ("carry", "static"))
        self.assertIn(("haul", "static"), earlier_stages("catch", "static"))
        self.assertEqual(later_stages("walk")[0], ("jump", "static"))
        self.assertEqual(later_stages("jump")[0], ("rough", "static"))
        self.assertEqual(later_stages("catch", "slow"), [("catch", "random"), ("catch", "noisy")])
        self.assertEqual(later_stages("catch", "noisy"), [])

    def test_rough_starts_from_walk_until_it_has_its_own_weights(self):
        import tempfile
        from pathlib import Path

        import torch

        from pick_and_carry.train_ppo import resolve_checkpoint

        def save(directory: Path, name: str, stage: str, gait: bool, obs: int = 69, act: int = 19, arm: str = "z1") -> None:
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"obs_dim": obs, "act_dim": act, "scripted_gait": gait, "arm": arm, "curriculum": stage},
                directory / name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            walk = stage_log_dir("walk", root=root)
            rough = stage_log_dir("rough", root=root)
            haul = stage_log_dir("haul", root=root)
            slow = stage_log_dir("catch", "slow", root=root)
            save(walk, "ppo_0001.pt", "walk", False, arm="panda")
            found = resolve_checkpoint(rough, "latest", 69, 19, False, "z1", "rough", root=root)
            self.assertIsNone(found)
            save(walk, "ppo_0002.pt", "walk", False)
            found = resolve_checkpoint(rough, "latest", 69, 19, False, "z1", "rough", root=root)
            self.assertEqual(found, walk / "ppo_0002.pt")
            save(rough, "ppo_0004.pt", "rough", False)
            found = resolve_checkpoint(rough, "latest", 69, 19, False, "z1", "rough", root=root)
            self.assertEqual(found, rough / "ppo_0004.pt")
            save(haul, "ppo_0001.pt", "haul", False)
            found = resolve_checkpoint(slow, "latest", 69, 19, False, "z1", "catch", "slow", root=root)
            self.assertEqual(found, haul / "ppo_0001.pt")
            found = resolve_checkpoint(
                walk, "latest", 69, 19, False, "z1", "walk", root=root, retreat=True
            )
            self.assertEqual(found, rough / "ppo_0004.pt")
            found = resolve_checkpoint(walk, "latest", 69, 19, False, "z1", "walk", root=root)
            self.assertEqual(found, walk / "ppo_0002.pt")


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

        from pick_and_carry.train_ppo import archive_incompatible_logs

        def save(directory: Path, name: str, obs: int) -> None:
            torch.save(
                {"obs_dim": obs, "act_dim": 19, "scripted_gait": False, "arm": "z1", "curriculum": "walk"},
                directory / name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertIsNone(archive_incompatible_logs(folder, 81, 19, False, "z1", "walk"))
            save(folder, "ppo_0100.pt", 69)
            (folder / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
            (folder / "events.out.tfevents.1").write_text("x", encoding="utf-8")
            moved = archive_incompatible_logs(folder, 81, 19, False, "z1", "walk")
            self.assertIsNotNone(moved)
            self.assertTrue((moved / "ppo_0100.pt").is_file())
            self.assertTrue((moved / "metrics.jsonl").is_file())
            self.assertTrue((moved / "events.out.tfevents.1").is_file())
            self.assertEqual(list(folder.glob("ppo_*.pt")), [])
            save(folder, "ppo_0001.pt", 81)
            self.assertIsNone(archive_incompatible_logs(folder, 81, 19, False, "z1", "walk"))
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
    def test_catch_without_weights_names_the_haul_folder(self):
        import tempfile
        from argparse import Namespace
        from pathlib import Path

        from pick_and_carry.config import EnvConfig
        from pick_and_carry.curriculum import stage_log_dir
        from pick_and_carry.train_ppo import missing_play_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = EnvConfig(curriculum="catch", catch_level="static", arm="z1", scripted_gait=True)
            args = Namespace(
                play=True,
                resume="latest",
                retreat=False,
                log_dir=stage_log_dir("catch", "static", "z1", root=root),
            )
            message = missing_play_checkpoint(args, cfg, root=root)
        self.assertIsNotNone(message)
        self.assertIn("再生できる重みがありません", message)
        self.assertIn("ppo_curriculum_haul", message)
        self.assertIn("ppo_z1", message)
