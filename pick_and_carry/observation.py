"""Pack the shared observation. Independent of Genesis.

Every curriculum stage uses this vector. Tasks that have no box or no speed
target leave those channels at zero. The object channel is the cube in the
base frame. With the wrist camera it is the detected centroid, or ``(0, 0, -1)``
when the cube is not in view.
"""

from __future__ import annotations

import torch

from pick_and_carry.helpers import world_to_base

LEG_VEL_SCALE = 0.05
ARM_VEL_SCALE = 0.05
GRIPPER_VEL_SCALE = 0.05
LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.25
UNSEEN_OBJECT = (0.0, 0.0, -1.0)


def object_in_base(
    obj_pos: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    *,
    use_wrist: bool,
    wrist_pos: torch.Tensor | None = None,
    wrist_visible: torch.Tensor | None = None,
) -> torch.Tensor:
    """Object position the policy sees, in the base frame."""
    if not use_wrist:
        return world_to_base(obj_pos, base_pos, base_quat)
    seen = world_to_base(wrist_pos, base_pos, base_quat)
    unseen = torch.zeros_like(seen)
    unseen[:, 0] = UNSEEN_OBJECT[0]
    unseen[:, 1] = UNSEEN_OBJECT[1]
    unseen[:, 2] = UNSEEN_OBJECT[2]
    return torch.where(wrist_visible[:, None], seen, unseen)


def pack_observation(
    *,
    leg_pos: torch.Tensor,
    leg_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    base_height: torch.Tensor,
    foot_contact: torch.Tensor,
    arm_pos: torch.Tensor,
    arm_vel: torch.Tensor,
    gripper_pos: torch.Tensor,
    gripper_vel: torch.Tensor,
    ee_pos: torch.Tensor,
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    grasped: torch.Tensor,
    phase: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Concatenate one observation row per environment. ``phase`` is an integer 0..3."""
    return torch.cat(
        (
            leg_pos,
            leg_vel * LEG_VEL_SCALE,
            projected_gravity,
            base_lin_vel * LIN_VEL_SCALE,
            base_ang_vel * ANG_VEL_SCALE,
            base_height,
            foot_contact,
            arm_pos,
            arm_vel * ARM_VEL_SCALE,
            gripper_pos,
            gripper_vel * GRIPPER_VEL_SCALE,
            ee_pos,
            object_pos,
            goal_pos,
            grasped.to(dtype=torch.float32).unsqueeze(-1),
            torch.nn.functional.one_hot(phase.long(), num_classes=4).to(dtype=torch.float32),
            command,
        ),
        dim=-1,
    )
