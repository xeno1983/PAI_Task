"""Curriculum rewards and layouts. No simulator."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.config import layout_dim
from pick_and_carry.curriculum import (
    SIT_HEIGHT,
    STAND_HEIGHT,
    earlier_stages,
    later_stages,
    locomotion_reward,
    manipulation_reward,
    pose_reward,
    posture_achieved,
    retreat_choice,
    sample_ball,
    sample_command,
    stage_layouts,
    stage_log_dir,
)


class LayoutTests(unittest.TestCase):
    def test_every_stage_uses_the_arm_layout(self):
        from pick_and_carry.arms import layouts

        shared = layouts(6)
        for stage in ("carry", "stand", "sit", "walk", "jump", "rough", "haul", "catch"):
            self.assertEqual(stage_layouts(stage, 6), shared)
        self.assertEqual(layout_dim(shared[0]), 69)
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


class RewardTests(unittest.TestCase):
    def test_matching_the_command_scores_higher(self):
        command = sample_command(1)
        matched, _, _ = locomotion_reward(
            base_lin_vel=command[:, :2].new_tensor([[command[0, 0], command[0, 1], 0.0]]),
            base_ang_vel=command.new_tensor([[0.0, 0.0, command[0, 2]]]),
            command=command,
            projected_gravity=command.new_tensor([[0.0, 0.0, -1.0]]),
            base_height=command.new_tensor([0.4]),
            actions=torch.zeros(1, 12),
            max_tilt_deg=60.0,
            min_base_height=0.18,
            dt=0.02,
        )
        missed, _, fallen = locomotion_reward(
            base_lin_vel=command.new_tensor([[0.0, 0.0, 0.0]]),
            base_ang_vel=torch.zeros(1, 3),
            command=command.new_tensor([[0.5, 0.0, 0.0]]),
            projected_gravity=command.new_tensor([[0.0, 0.0, -1.0]]),
            base_height=command.new_tensor([0.4]),
            actions=torch.zeros(1, 12),
            max_tilt_deg=60.0,
            min_base_height=0.18,
            dt=0.02,
        )
        self.assertGreater(matched.item(), missed.item())
        self.assertFalse(bool(fallen))

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
