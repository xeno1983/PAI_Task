"""Joint targets from the policy action. Independent of Genesis.

``split`` walks with the legs, then holds them while the arm picks up or releases.
``walk`` commands the legs the whole episode and locks the arm open.
``both`` commands the legs and the arm together.
"""

from __future__ import annotations

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
    """Reference pose plus the policy, held still during pickup and release in split mode."""
    pose = default_leg.expand(leg_act.shape[0], -1)
    commanded = (pose + leg_act * action_scale).clamp(lower, upper)
    if control_split in ("walk", "both"):
        return commanded
    return torch.where(walk.unsqueeze(-1), commanded, pose)


def gripper_open_amount(joint: torch.Tensor, joint_open: float, joint_closed: float) -> torch.Tensor:
    """How far the gripper is from closed. Zero is closed. The Z1 joint is in radians."""
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
    Action -1 closes and action +1 opens.
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

