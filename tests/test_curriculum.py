"""Curriculum rewards and layouts. No simulator."""

from __future__ import annotations

import unittest

import torch

from pick_and_carry.config import layout_dim
from pick_and_carry.curriculum import (
    decode_base_action,
    earlier_stages,
    locomotion_reward,
    manipulation_reward,
    pack_walk_observation,
    sample_ball,
    sample_command,
    stage_layouts,
    stage_log_dir,
)


class LayoutTests(unittest.TestCase):
    def test_carry_keeps_the_existing_layout(self):
        self.assertIsNone(stage_layouts("carry", 7))

    def test_walk_is_legs_and_a_command(self):
        observation, action = stage_layouts("walk", 7)
        self.assertEqual(layout_dim(observation), 41)
        self.assertEqual(layout_dim(action), 12)
        self.assertEqual(stage_layouts("rough", 7), (observation, action))

    def test_catch_adds_the_ball_and_a_body_command(self):
        observation, action = stage_layouts("catch", 7)
        self.assertEqual(layout_dim(observation), 67)
        self.assertEqual(layout_dim(action), 11)
        self.assertEqual(layout_dim(stage_layouts("catch", 6)[1]), 10)


class CommandTests(unittest.TestCase):
    def test_base_action_stays_inside_the_gait(self):
        raw = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
        command = decode_base_action(raw)
        self.assertTrue(torch.all(command[:, 0].abs() <= 0.45 + 1e-5))
        self.assertTrue(torch.all(command[:, 1].abs() <= 0.25 + 1e-5))
        self.assertTrue(torch.all(command[:, 2].abs() <= 0.80 + 1e-5))

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
        observation, _ = stage_layouts("walk", 7)
        obs = pack_walk_observation(
            leg_pos=torch.zeros(2, 12),
            leg_vel=torch.zeros(2, 12),
            projected_gravity=torch.zeros(2, 3),
            base_lin_vel=torch.zeros(2, 3),
            base_ang_vel=torch.zeros(2, 3),
            base_height=torch.zeros(2, 1),
            foot_contact=torch.zeros(2, 4),
            command=torch.zeros(2, 3),
        )
        self.assertEqual(obs.shape, (2, layout_dim(observation)))


class TransferTests(unittest.TestCase):
    def test_each_stage_points_at_the_previous_skill(self):
        self.assertEqual(earlier_stages("walk"), [])
        self.assertEqual(earlier_stages("rough"), [("walk", "static")])
        self.assertEqual(earlier_stages("haul"), [])
        self.assertEqual(earlier_stages("catch", "static"), [("haul", "static")])
        self.assertEqual(
            earlier_stages("catch", "noisy"),
            [("catch", "random"), ("catch", "slow"), ("catch", "static"), ("haul", "static")],
        )

    def test_rough_starts_from_walk_until_it_has_its_own_weights(self):
        import tempfile
        from pathlib import Path

        import torch

        from pick_and_carry.train_ppo import find_locomotion_checkpoint, resolve_checkpoint

        def save(directory: Path, name: str, stage: str, gait: bool, obs: int = 41, act: int = 12) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"obs_dim": obs, "act_dim": act, "scripted_gait": gait, "arm": "panda", "curriculum": stage},
                directory / name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            walk = stage_log_dir("walk", root=root)
            rough = stage_log_dir("rough", root=root)
            haul = stage_log_dir("haul", root=root)
            slow = stage_log_dir("catch", "slow", root=root)
            save(walk, "ppo_0002.pt", "walk", False)
            found = resolve_checkpoint(rough, "latest", 41, 12, False, "z1", "rough", root=root)
            self.assertEqual(found, walk / "ppo_0002.pt")
            save(rough, "ppo_0004.pt", "rough", False)
            found = resolve_checkpoint(rough, "latest", 41, 12, False, "panda", "rough", root=root)
            self.assertEqual(found, rough / "ppo_0004.pt")
            save(haul, "ppo_0001.pt", "haul", True, obs=67, act=11)
            found = resolve_checkpoint(slow, "latest", 67, 11, True, "panda", "catch", "slow", root=root)
            self.assertEqual(found, haul / "ppo_0001.pt")
            legs = find_locomotion_checkpoint(root)
            self.assertEqual(legs, rough / "ppo_0004.pt")
