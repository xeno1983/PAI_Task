"""Phase-gated reward. Independent of Genesis so the terms can be unit-tested.

`compute_stage_reward` builds every term. `task_reward` keeps a term only in
the phase that should learn it, then adds the walking-speed term.
"""

from __future__ import annotations

import torch

from pick_and_carry.config import EnvConfig
from pick_and_carry.helpers import compute_stage_reward, fallen_mask, pairwise_distance, place_mask

# Progress is credited only up to this many metres in one step, so a reset
# cannot look like a huge approach.
PROGRESS_STEP_CAP = 0.05
# Learned gait: track this forward speed. Standing still scores well below it.
TRACK_SPEED = 0.40
TRACK_SIGMA = 0.30
SLIP_COEF = 0.4
WALK_SPEED_COEF = 6.0
SPLIT_SPEED_COEF = 4.0

_WALK_ONLY_ZERO = (
    "approach_ee",
    "approach_base",
    "grasp_shape",
    "grasp_bonus",
    "transport",
    "place",
    "progress_ee",
    "progress_goal",
    "progress_base",
)


def task_reward(
    *,
    ee_pos: torch.Tensor,
    obj_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    base_pos: torch.Tensor,
    grasped: torch.Tensor,
    prev_grasped: torch.Tensor,
    was_grasped: torch.Tensor,
    place_latched: torch.Tensor,
    projected_gravity: torch.Tensor,
    phase: torch.Tensor,
    prev_ee_dist: torch.Tensor,
    prev_goal_dist: torch.Tensor,
    prev_base_dist: torch.Tensor,
    base_obj_dist: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    gripper_opening: torch.Tensor,
    actions: torch.Tensor,
    cfg: EnvConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reward, terms, newly placed, fallen, and the updated latches and distances.

    ``ee_dist`` is logged in ``terms`` and is not added into the reward.
    """
    fallen = fallen_mask(projected_gravity, base_pos[:, 2], cfg.max_tilt_deg, cfg.min_base_height)
    placed_now = place_mask(obj_pos, goal_pos, grasped, was_grasped, cfg.goal_radius, cfg.place_height)
    newly_placed = placed_now & ~place_latched
    place_latched = place_latched | placed_now
    newly_grasped = grasped & ~prev_grasped
    reward, terms = compute_stage_reward(
        ee_pos=ee_pos,
        obj_pos=obj_pos,
        goal_pos=goal_pos,
        base_pos=base_pos,
        grasped=grasped,
        newly_grasped=newly_grasped,
        placed=newly_placed,
        fallen=fallen,
        gripper_opening=gripper_opening,
        gripper_open_width=cfg.gripper_open,
        actions=actions,
        scales=cfg.reward_scales,
        dt=cfg.dt,
        approach_sigma=cfg.approach_sigma,
        transport_sigma=cfg.transport_sigma,
        base_sigma=cfg.base_sigma,
    )
    dist_ee = pairwise_distance(ee_pos, obj_pos)
    dist_goal = pairwise_distance(obj_pos, goal_pos)
    walk_object = (phase == 0).to(dtype=reward.dtype)
    pickup = (phase == 1).to(dtype=reward.dtype)
    walk_goal = (phase == 2).to(dtype=reward.dtype)
    release = (phase == 3).to(dtype=reward.dtype)
    terms["approach_ee"] = terms["approach_ee"] * pickup
    terms["grasp_shape"] = terms["grasp_shape"] * pickup
    terms["approach_base"] = terms["approach_base"] * walk_object
    terms["transport"] = terms["transport"] * walk_goal
    terms["progress_ee"] = cfg.reward_scales.progress_ee * (prev_ee_dist - dist_ee).clamp(-PROGRESS_STEP_CAP, PROGRESS_STEP_CAP) * pickup
    terms["progress_goal"] = cfg.reward_scales.progress_goal * (prev_goal_dist - dist_goal).clamp(-PROGRESS_STEP_CAP, PROGRESS_STEP_CAP) * walk_goal
    terms["progress_base"] = cfg.reward_scales.progress_base * (prev_base_dist - base_obj_dist).clamp(-PROGRESS_STEP_CAP, PROGRESS_STEP_CAP) * walk_object
    walk = walk_object + walk_goal
    if cfg.control_split == "walk":
        for name in _WALK_ONLY_ZERO:
            if name in terms:
                terms[name] = torch.zeros_like(terms[name])
        walk = (~fallen).to(dtype=reward.dtype)
    if cfg.scripted_gait:
        forward = base_lin_vel[:, 0].clamp(-0.5, 1.0)
        terms["forward"] = cfg.reward_scales.forward * forward * walk * cfg.dt
    else:
        vx = base_lin_vel[:, 0]
        vy = base_lin_vel[:, 1]
        yaw = base_ang_vel[:, 2]
        track = torch.exp(-((vx - TRACK_SPEED) / TRACK_SIGMA) ** 2)
        slip = vy.square() + (0.5 * yaw).square()
        speed_coef = WALK_SPEED_COEF if cfg.control_split == "walk" else SPLIT_SPEED_COEF
        terms["forward"] = (speed_coef * track - SLIP_COEF * slip) * walk * cfg.dt
    open_frac = (gripper_opening / cfg.gripper_open).clamp(0.0, 1.0)
    if cfg.control_split == "walk":
        terms["release"] = torch.zeros_like(reward)
    else:
        terms["release"] = cfg.reward_scales.release * open_frac * release * cfg.dt
    terms["ee_dist"] = dist_ee
    reward = torch.stack([value for name, value in terms.items() if name != "ee_dist"], dim=0).sum(dim=0)
    return (
        reward,
        terms,
        newly_placed,
        fallen,
        place_latched,
        dist_ee.detach(),
        dist_goal.detach(),
        base_obj_dist.detach(),
    )
