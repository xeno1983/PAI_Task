"""Reward stages, quaternion frames, and observation layout."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from pick_and_carry.config import NUM_ACTIONS, NUM_OBS, OBS_LAYOUT, EnvConfig, RewardScales, layout_dim
from pick_and_carry.control import arm_gripper_command, learned_leg_target
from pick_and_carry.observation import pack_observation
from pick_and_carry.reward import task_reward
from scripts.infer_gui import canvas_point, world_point

from pick_and_carry.helpers import (
    compute_stage_reward,
    detect_wrist_target,
    fallen_mask,
    grasp_mask,
    place_mask,
    quat_apply,
    task_phase,
    world_to_base,
    yaw_quat,
)


class LayoutTests(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(layout_dim(OBS_LAYOUT), 69)
        self.assertEqual(NUM_OBS, 69)
        self.assertEqual(NUM_ACTIONS, 19)


class GeometryTests(unittest.TestCase):
    def test_yaw_rotation(self):
        q = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
        rotated = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]]))
        self.assertTrue(torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5))

    def test_world_to_base_identity(self):
        points = torch.tensor([[1.0, 2.0, 3.0]])
        base = torch.zeros(1, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(world_to_base(points, base, quat), points))

    def test_yaw_quat_norm(self):
        q = yaw_quat(torch.tensor([0.3, -1.2]))
        self.assertTrue(torch.allclose(q.norm(dim=-1), torch.ones(2), atol=1e-5))


class RewardTests(unittest.TestCase):
    def test_stage_bonuses(self):
        n = 3
        ee = torch.zeros(n, 3)
        obj = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.02], [0.0, 0.0, 0.0]])
        goal = torch.zeros(n, 3)
        goal[:, 2] = 0.02
        base = torch.zeros(n, 3)
        grasped = torch.tensor([False, False, True])
        newly = torch.tensor([True, False, False])
        placed = torch.tensor([False, True, False])
        fallen = torch.tensor([False, False, True])
        scales = RewardScales()
        reward, terms = compute_stage_reward(
            ee_pos=ee,
            obj_pos=obj,
            goal_pos=goal,
            base_pos=base,
            grasped=grasped,
            newly_grasped=newly,
            placed=placed,
            fallen=fallen,
            gripper_opening=torch.full((n,), 0.04),
            gripper_open_width=0.04,
            actions=torch.zeros(n, NUM_ACTIONS),
            scales=scales,
            dt=0.02,
            approach_sigma=0.25,
            transport_sigma=0.4,
            base_sigma=0.6,
        )
        self.assertAlmostEqual(float(terms["grasp_bonus"][0]), scales.grasp_bonus)
        self.assertAlmostEqual(float(terms["place"][1]), scales.place_bonus)
        self.assertAlmostEqual(float(terms["fall"][2]), -scales.fall)
        self.assertEqual(reward.shape, (n,))
        self.assertGreater(float(terms["transport"][2]), 0.0)
        self.assertEqual(float(terms["approach_ee"][2]), 0.0)

    def test_grasp_and_place_masks(self):
        ee = torch.tensor([[0.0, 0.0, 0.1]])
        obj = torch.tensor([[0.0, 0.0, 0.12]])
        goal = torch.tensor([[0.0, 0.0, 0.02]])
        self.assertTrue(bool(grasp_mask(ee, obj, torch.tensor([0.0]), 0.1, 0.012)))
        self.assertFalse(bool(grasp_mask(ee, obj, torch.tensor([0.04]), 0.1, 0.012)))
        grasped = torch.tensor([False])
        was = torch.tensor([True])
        low_obj = torch.tensor([[0.05, 0.0, 0.03]])
        self.assertTrue(bool(place_mask(low_obj, goal, grasped, was, 0.15, 0.08)))
        upright = torch.tensor([[0.0, 0.0, -1.0]])
        height = torch.tensor([0.42])
        self.assertFalse(bool(fallen_mask(upright, height, 60.0, 0.18)))
        upside_down = torch.tensor([[0.0, 0.0, 1.0]])
        self.assertTrue(bool(fallen_mask(upside_down, height, 60.0, 0.18)))

    def test_walk_pick_walk_release_phases(self):
        grasped = torch.tensor([False, False, True, True])
        base_obj = torch.tensor([1.2, 0.4, 0.3, 0.3])
        obj_goal = torch.tensor([2.0, 1.5, 1.2, 0.2])
        phase = task_phase(grasped, base_obj, obj_goal, 0.55, 0.45)
        self.assertEqual(phase.tolist(), [0, 1, 2, 3])

    def test_wrist_camera_centroid_uses_only_the_target_pixels(self):
        depth = np.full((8, 8), 0.5, dtype=np.float32)
        seg = np.zeros((8, 8), dtype=np.int32)
        seg[3:5, 3:5] = 7
        depth[3:5, 3:5] = 0.20
        pose = np.eye(4, dtype=np.float32)
        positions, visible = detect_wrist_target(depth, seg, [7], focal=8.0, cx=4.0, cy=4.0, transform=pose)
        self.assertTrue(bool(visible[0]))
        self.assertAlmostEqual(float(positions[0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(positions[0, 2]), -0.20, places=4)
        missed, unseen = detect_wrist_target(depth, seg, [3], focal=8.0, cx=4.0, cy=4.0, transform=pose)
        self.assertFalse(bool(unseen[0]))
        self.assertAlmostEqual(float(missed[0, 0]), 0.0, places=4)

    def test_walk_mode_drops_manipulation_reward(self):
        n = 1
        cfg = EnvConfig(control_split="walk", scripted_gait=False)
        zeros = torch.zeros(n, 3)
        reward, terms, *_ = task_reward(
            ee_pos=zeros,
            obj_pos=zeros,
            goal_pos=zeros,
            base_pos=torch.tensor([[0.0, 0.0, 0.42]]),
            grasped=torch.tensor([True]),
            prev_grasped=torch.tensor([False]),
            was_grasped=torch.tensor([True]),
            place_latched=torch.tensor([False]),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            phase=torch.tensor([1]),
            prev_ee_dist=torch.tensor([0.2]),
            prev_goal_dist=torch.tensor([1.0]),
            prev_base_dist=torch.tensor([1.0]),
            base_obj_dist=torch.tensor([1.0]),
            base_lin_vel=torch.tensor([[0.40, 0.0, 0.0]]),
            base_ang_vel=torch.zeros(n, 3),
            gripper_opening=torch.tensor([0.0]),
            actions=torch.zeros(n, NUM_ACTIONS),
            cfg=cfg,
        )
        self.assertEqual(float(terms["grasp_bonus"]), 0.0)
        self.assertEqual(float(terms["place"]), 0.0)
        self.assertGreater(float(terms["forward"]), 0.0)
        self.assertEqual(reward.shape, (n,))

    def test_observation_scales_and_unseen_object(self):
        obs = pack_observation(
            leg_pos=torch.zeros(1, 12),
            leg_vel=torch.ones(1, 12),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            base_lin_vel=torch.ones(1, 3),
            base_ang_vel=torch.ones(1, 3),
            base_height=torch.tensor([[0.42]]),
            foot_contact=torch.zeros(1, 4),
            arm_pos=torch.zeros(1, 6),
            arm_vel=torch.ones(1, 6),
            gripper_pos=torch.tensor([[0.04]]),
            gripper_vel=torch.ones(1, 1),
            ee_pos=torch.zeros(1, 3),
            object_pos=torch.tensor([[0.0, 0.0, -1.0]]),
            goal_pos=torch.zeros(1, 3),
            grasped=torch.tensor([False]),
            phase=torch.tensor([2]),
            command=torch.zeros(1, 3),
        )
        self.assertEqual(obs.shape, (1, NUM_OBS))
        self.assertAlmostEqual(float(obs[0, 12]), 0.05)
        self.assertAlmostEqual(float(obs[0, 27]), 2.0)
        self.assertAlmostEqual(float(obs[0, 30]), 0.25)
        self.assertEqual(obs[0, 62:66].tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_split_holds_the_arm_while_walking(self):
        arm_act = torch.ones(1, 7)
        grip = torch.tensor([[-1.0]])
        default = torch.zeros(1, 7)
        lower = torch.full((7,), -3.0)
        upper = torch.full((7,), 3.0)
        walking, opening = arm_gripper_command(
            arm_act, grip, default, torch.tensor([0]), control_split="split", arm_scale=1.2, gripper_open=0.04, lower=lower, upper=upper
        )
        self.assertTrue(torch.equal(walking, default))
        self.assertAlmostEqual(float(opening), 0.04)
        picking, _ = arm_gripper_command(
            arm_act, grip, default, torch.tensor([1]), control_split="split", arm_scale=1.2, gripper_open=0.04, lower=lower, upper=upper
        )
        self.assertTrue(torch.allclose(picking, torch.full((1, 7), 1.2)))
        legs = learned_leg_target(
            torch.ones(1, 12),
            torch.zeros(12),
            torch.tensor([False]),
            control_split="split",
            action_scale=0.5,
            lower=torch.full((12,), -3.0),
            upper=torch.full((12,), 3.0),
        )
        self.assertTrue(torch.equal(legs, torch.zeros(1, 12)))

    def test_click_map_round_trips_world_xy(self):
        for point in ((0.0, 0.0), (1.75, -0.2), (4.0, 0.4)):
            px, py = canvas_point(*point)
            back = world_point(px, py)
            self.assertAlmostEqual(back[0], point[0], places=4)
            self.assertAlmostEqual(back[1], point[1], places=4)


if __name__ == "__main__":
    unittest.main()
