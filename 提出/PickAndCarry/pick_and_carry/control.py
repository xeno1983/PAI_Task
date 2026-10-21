"""Joint targets from the policy action. Independent of Genesis.

Legs: CPG expansion happens in the env; this module only maps residuals.

Arm / gripper (manipulation stages): the environment supplies an IK / phase
baseline; the policy outputs a small residual that is scaled and added.

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


def arm_residual_command(
    q_ik: torch.Tensor,
    arm_act: torch.Tensor,
    *,
    arm_scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """``q = clamp(q_ik + arm_scale * Δq)`` with policy action roughly in [-1, 1]."""
    return (q_ik + arm_act * arm_scale).clamp(lower, upper)


def gripper_baseline(
    phase: torch.Tensor,
    *,
    curriculum: str,
    gripper_open: float,
    gripper_closed: float,
    ee_near: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scripted open/close prior. Shape ``(N,)`` joint positions.

    Carry / haul follow pick-and-place phases. Reach / catch open by default and
    close when ``ee_near`` is true (optional proximity flag).
    """
    n = phase.shape[0]
    device = phase.device
    open_t = torch.full((n,), float(gripper_open), device=device)
    closed_t = torch.full((n,), float(gripper_closed), device=device)
    mid = 0.5 * (gripper_open + gripper_closed)
    mid_t = torch.full((n,), float(mid), device=device)

    if curriculum in ("reach", "catch"):
        base = open_t.clone()
        if ee_near is not None:
            base = torch.where(ee_near, closed_t, base)
        return base

    # phase 0 walk-to-object: open; 1 pick: closing; 2 carry: closed; 3 place: open
    base = open_t.clone()
    base = torch.where(phase == 1, mid_t, base)
    base = torch.where(phase == 2, closed_t, base)
    base = torch.where(phase == 3, open_t, base)
    return base


def gripper_residual_command(
    g_base: torch.Tensor,
    grip_act: torch.Tensor,
    *,
    residual_scale: float,
    gripper_open: float,
    gripper_closed: float,
) -> torch.Tensor:
    """``g = clamp(g_base + residual_scale * Δg)`` within the finger joint range."""
    delta = grip_act.squeeze(-1) * residual_scale
    lo = min(gripper_open, gripper_closed)
    hi = max(gripper_open, gripper_closed)
    return (g_base + delta).clamp(lo, hi)


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
    q_ik: torch.Tensor | None = None,
    grip_residual_scale: float | None = None,
    curriculum: str = "carry",
    ee_near: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arm joint targets and the gripper joint position.

    When ``q_ik`` is given (manipulation stages), arm/gripper use IK + residual.
    Otherwise falls back to the legacy default-pose + full-scale action mapping
    (kept for tests and non-IK call sites).
    """
    if q_ik is not None:
        arm_target = arm_residual_command(q_ik, arm_act, arm_scale=arm_scale, lower=lower, upper=upper)
        g_base = gripper_baseline(
            phase,
            curriculum=curriculum,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            ee_near=ee_near,
        )
        # Dedicated grip residual; default ≈ 25% of the open–close span.
        g_scale = (
            float(grip_residual_scale)
            if grip_residual_scale is not None
            else 0.25 * abs(gripper_open - gripper_closed)
        )
        opening = gripper_residual_command(
            g_base,
            grip_act,
            residual_scale=g_scale,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
        )
        return arm_target, opening

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
