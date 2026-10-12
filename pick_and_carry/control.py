"""Joint targets from the policy action. Independent of Genesis.

``split`` walks with the legs, then holds them while the arm picks up or releases.
``walk`` commands the legs the whole episode and locks the arm open.
``both`` commands the legs and the arm together.
The scripted trot is only used when ``scripted_gait`` is set.
"""

from __future__ import annotations

import math

import torch


def learned_leg_target(
    leg_act: torch.Tensor,
    default_leg: torch.Tensor,
    walk: torch.Tensor,
    *,
    control_split: str,
    action_scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Standing pose plus the policy, held still during pickup and release in split mode."""
    pose = default_leg.expand(leg_act.shape[0], -1)
    commanded = (pose + leg_act * action_scale).clamp(lower, upper)
    if control_split in ("walk", "both"):
        return commanded
    return torch.where(walk.unsqueeze(-1), commanded, pose)


def gripper_open_amount(joint: torch.Tensor, joint_open: float, joint_closed: float) -> torch.Tensor:
    """How far the gripper is from closed. Zero is closed. Panda is metres, Z1 is radians."""
    direction = 1.0 if joint_open >= joint_closed else -1.0
    return (joint - joint_closed) * direction


def arm_gripper_command(
    arm_act: torch.Tensor,
    grip_act: torch.Tensor,
    default_arm: torch.Tensor,
    phase: torch.Tensor,
    *,
    control_split: str,
    arm_scale: float,
    gripper_open: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    gripper_closed: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arm joint targets and the gripper joint position.

    ``gripper_open`` and ``gripper_closed`` are joint positions, not the reward width.
    Action -1 closes and action +1 opens. For the Panda both fingers use metres and
    closed is 0, so this matches the previous mapping.
    """
    walk = (phase == 0) | (phase == 2)
    arm_target = (default_arm + arm_act * arm_scale).clamp(lower, upper)
    span = gripper_open - gripper_closed
    opening = (gripper_closed + (grip_act + 1.0) * 0.5 * span).squeeze(-1)
    if control_split == "walk":
        return default_arm.expand_as(arm_target), torch.full_like(opening, gripper_open)
    if control_split != "both":
        arm_target = torch.where(walk.unsqueeze(-1), default_arm.expand_as(arm_target), arm_target)
        opening = torch.where(phase == 0, torch.full_like(opening, gripper_open), opening)
        opening = torch.where(phase == 2, torch.full_like(opening, gripper_closed), opening)
    return arm_target, opening


def scripted_trot_pose(
    leg_act: torch.Tensor,
    default_leg: torch.Tensor,
    walk: torch.Tensor,
    gait_blend: torch.Tensor,
    gait_time: torch.Tensor,
    *,
    dt: float,
    gait_period: float,
    action_scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    yaw_err: torch.Tensor | None = None,
    speed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagonal trot blended in over one second. Returns the pose and the updated gait state.

    ``yaw_err`` is set only while carrying supplies around a yard. Positive means
    the body is aimed to the left of the target.
    """
    n = leg_act.shape[0]
    pose = default_leg.expand(n, -1).clone()
    rate = dt / 1.00
    want = walk.to(dtype=gait_blend.dtype)
    gait_blend = (gait_blend + (want - gait_blend).clamp(-rate, rate)).clamp(0.0, 1.0)
    gait_time = gait_time + dt
    offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), dtype=torch.float32, device=leg_act.device)
    phase = (gait_time / gait_period).unsqueeze(-1) + offsets
    phase = phase - torch.floor(phase)
    swing = phase < 0.4
    swing_u = (phase / 0.4).clamp(0.0, 1.0)
    stance_u = ((phase - 0.4) / 0.6).clamp(0.0, 1.0)
    thigh = torch.where(swing, 0.98 + (0.62 - 0.98) * swing_u, 0.62 + (0.98 - 0.62) * stance_u)
    calf = torch.where(swing, -1.5 - 0.40 * torch.sin(math.pi * swing_u), torch.full_like(phase, -1.5))
    mix = gait_blend.unsqueeze(-1)
    if speed is not None:
        mix = mix * speed.to(dtype=mix.dtype).unsqueeze(-1).clamp(0.0, 1.0)
    pose[:, 1::3] = pose[:, 1::3] * (1.0 - mix) + thigh * mix
    pose[:, 2::3] = pose[:, 2::3] * (1.0 - mix) + calf * mix
    if yaw_err is not None:
        turn = ((yaw_err.abs() - 0.35) / 0.7).clamp(0.0, 1.0).unsqueeze(-1)
        stand = default_leg.expand(n, -1)
        pose[:, 1::3] = pose[:, 1::3] * (1.0 - 0.40 * turn) + stand[:, 1::3] * (0.40 * turn)
        steer = (-1.3 * yaw_err).clamp(-0.22, 0.22) * gait_blend
        pose[:, 1] = pose[:, 1] - steer
        pose[:, 7] = pose[:, 7] - steer
        pose[:, 4] = pose[:, 4] + steer
        pose[:, 10] = pose[:, 10] + steer
    residual = leg_act * action_scale * 0.25 * mix.to(dtype=leg_act.dtype)
    return (pose + residual).clamp(lower, upper), gait_blend, gait_time
