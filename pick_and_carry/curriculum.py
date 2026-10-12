"""Staged tasks on the same robot. The pick-and-carry observation stays unchanged.

walk learns the legs. rough continues those weights on bumps. haul and catch
reuse that walk as a frozen leg controller and train the arm plus a body
command. Each catch level continues the previous level, then haul.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pick_and_carry.helpers import fallen_mask, world_to_base

STAGES = ("carry", "walk", "rough", "haul", "catch")
CATCH_LEVELS = ("static", "slow", "random", "noisy")

WALK_COMMAND_SECONDS = 4.0
CATCH_HOLD_SECONDS = 1.0
BALL_RADIUS = 0.035
PROGRESS_CAP = 0.05


def earlier_stages(stage: str, catch_level: str = "static") -> list[tuple[str, str]]:
    """Same-sized policies that can initialize this stage, nearest first."""
    if stage == "rough":
        return [("walk", "static")]
    if stage != "catch":
        return []
    if catch_level not in CATCH_LEVELS:
        raise ValueError(f"unknown catch level {catch_level}")
    index = CATCH_LEVELS.index(catch_level)
    priors = [("catch", level) for level in reversed(CATCH_LEVELS[:index])]
    priors.append(("haul", "static"))
    return priors


def stage_log_dir(stage: str, catch_level: str = "static", arm: str = "panda", root: Path | None = None) -> Path:
    """Folder used by the launcher for this stage. Walk and rough ignore the arm."""
    if stage == "carry":
        relative = "logs/ppo_z1" if arm == "z1" else "logs/ppo_walk"
    elif stage == "walk":
        relative = "logs/ppo_curriculum_walk"
    elif stage == "rough":
        relative = "logs/ppo_curriculum_rough"
    elif stage == "haul":
        relative = "logs/ppo_curriculum_haul"
    elif stage == "catch":
        if catch_level not in CATCH_LEVELS:
            raise ValueError(f"unknown catch level {catch_level}")
        relative = f"logs/ppo_curriculum_catch_{catch_level}"
    else:
        raise ValueError(f"unknown curriculum stage {stage}")
    base = Path(root) if root is not None else Path()
    return base / relative


def stage_layouts(stage: str, arm_dof: int) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]] | None:
    """Observation and action layout for a stage. None keeps the pick-and-carry layout."""
    if stage == "carry":
        return None
    if stage in ("walk", "rough"):
        observation = (
            ("leg_pos", 12),
            ("leg_vel", 12),
            ("projected_gravity", 3),
            ("base_lin_vel", 3),
            ("base_ang_vel", 3),
            ("base_height", 1),
            ("foot_contact", 4),
            ("command", 3),
        )
        return observation, (("leg", 12),)
    if stage not in ("haul", "catch"):
        raise ValueError(f"unknown curriculum stage {stage}")
    observation = (
        ("leg_pos", 12),
        ("leg_vel", 12),
        ("projected_gravity", 3),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("base_height", 1),
        ("foot_contact", 4),
        ("arm_pos", arm_dof),
        ("arm_vel", arm_dof),
        ("gripper_pos", 1),
        ("gripper_vel", 1),
        ("ee_pos", 3),
        ("object_pos", 3),
        ("object_vel", 3),
        ("command", 3),
        ("grasped", 1),
    )
    return observation, (("arm", arm_dof), ("gripper", 1), ("base", 3))


def sample_command(count: int, device: torch.device | None = None) -> torch.Tensor:
    """Body-frame targets: forward m/s, left m/s, yaw rad/s."""
    if device is None:
        device = torch.device("cpu")
    forward = torch.empty(count, device=device).uniform_(-0.25, 0.50)
    left = torch.empty(count, device=device).uniform_(-0.20, 0.20)
    yaw = torch.empty(count, device=device).uniform_(-0.60, 0.60)
    return torch.stack((forward, left, yaw), dim=-1)


def decode_base_action(raw: torch.Tensor) -> torch.Tensor:
    """Map policy outputs in [-1, 1] to forward, left, and yaw rates."""
    scales = raw.new_tensor((0.45, 0.25, 0.80))
    return raw.clamp(-1.0, 1.0) * scales


def sample_ball(
    level: str,
    base_pos: torch.Tensor,
    *,
    radius: float = BALL_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World position, linear velocity, and roll rate for one catch level."""
    count = base_pos.shape[0]
    device = base_pos.device
    position = base_pos.clone()
    position[:, 2] = radius
    linear = torch.zeros((count, 3), device=device)
    if level == "static":
        position[:, 0] = base_pos[:, 0] + 0.40
        position[:, 1] = base_pos[:, 1]
    elif level == "slow":
        position[:, 0] = base_pos[:, 0] + 1.15
        position[:, 1] = base_pos[:, 1] + torch.empty(count, device=device).uniform_(-0.10, 0.10)
        linear[:, 0] = -0.28
    elif level in ("random", "noisy"):
        distance = torch.empty(count, device=device).uniform_(0.70, 1.50)
        angle = torch.empty(count, device=device).uniform_(-0.90, 0.90)
        position[:, 0] = base_pos[:, 0] + distance * torch.cos(angle)
        position[:, 1] = base_pos[:, 1] + distance * torch.sin(angle)
        toward = base_pos[:, :2] - position[:, :2]
        toward = toward / torch.linalg.norm(toward, dim=-1, keepdim=True).clamp(min=1e-4)
        heading = torch.atan2(toward[:, 1], toward[:, 0])
        heading = heading + torch.empty(count, device=device).uniform_(-0.30, 0.30)
        speed = torch.empty(count, device=device).uniform_(0.15, 0.55)
        linear[:, 0] = speed * torch.cos(heading)
        linear[:, 1] = speed * torch.sin(heading)
    else:
        raise ValueError(f"unknown catch level {level}")
    angular = torch.stack(
        (linear[:, 1] / radius, -linear[:, 0] / radius, torch.zeros(count, device=device)),
        dim=-1,
    )
    return position, linear, angular


def obscure_motion(position: torch.Tensor, velocity: torch.Tensor, level: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 6 starts here: the policy sees a noisy position instead of the exact one."""
    if level != "noisy":
        return position, velocity
    return position + torch.randn_like(position) * 0.03, velocity + torch.randn_like(velocity) * 0.05


def apply_lateral(pose: torch.Tensor, left_speed: torch.Tensor) -> torch.Tensor:
    """Small hip bias so a sideways command is not only a reward."""
    steered = pose.clone()
    lean = (left_speed * 0.35).clamp(-0.18, 0.18)
    steered[:, 0] = steered[:, 0] + lean
    steered[:, 3] = steered[:, 3] - lean
    steered[:, 6] = steered[:, 6] + lean
    steered[:, 9] = steered[:, 9] - lean
    return steered


def pack_walk_observation(
    *,
    leg_pos: torch.Tensor,
    leg_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    base_height: torch.Tensor,
    foot_contact: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            leg_pos,
            leg_vel * 0.05,
            projected_gravity,
            base_lin_vel * 2.0,
            base_ang_vel * 0.25,
            base_height,
            foot_contact,
            command,
        ),
        dim=-1,
    )


def pack_body_observation(
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
    object_vel: torch.Tensor,
    command: torch.Tensor,
    grasped: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            leg_pos,
            leg_vel * 0.05,
            projected_gravity,
            base_lin_vel * 2.0,
            base_ang_vel * 0.25,
            base_height,
            foot_contact,
            arm_pos,
            arm_vel * 0.05,
            gripper_pos,
            gripper_vel * 0.05,
            ee_pos,
            object_pos,
            object_vel * 0.5,
            command,
            grasped.to(dtype=torch.float32).unsqueeze(-1),
        ),
        dim=-1,
    )


def locomotion_reward(
    *,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    command: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    actions: torch.Tensor,
    max_tilt_deg: float,
    min_base_height: float,
    dt: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Track a speed command. Falling ends the positive terms."""
    fallen = fallen_mask(projected_gravity, base_height, max_tilt_deg, min_base_height)
    upright = (~fallen).to(dtype=base_lin_vel.dtype)
    error = (base_lin_vel[:, 0] - command[:, 0]).square()
    error = error + (base_lin_vel[:, 1] - command[:, 1]).square()
    error = error + (base_ang_vel[:, 2] - command[:, 2]).square()
    track = torch.exp(-error / 0.25)
    alive = upright * 0.4 * dt
    effort = -0.01 * actions.square().sum(dim=-1) * dt
    fall = -5.0 * fallen.to(dtype=base_lin_vel.dtype)
    terms = {"track": 4.0 * track * upright * dt, "alive": alive, "effort": effort, "fall": fall}
    reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    terms["tracking"] = track.detach()
    return reward, terms, fallen


def manipulation_reward(
    *,
    stage: str,
    ee_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_vel: torch.Tensor,
    base_lin_vel_world: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    command: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    grasped: torch.Tensor,
    closed: torch.Tensor,
    actions: torch.Tensor,
    prev_distance: torch.Tensor,
    max_tilt_deg: float,
    min_base_height: float,
    dt: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hold a carried box, or meet a rolling ball. Returns reward, terms, caught, fallen, distance."""
    fallen = fallen_mask(projected_gravity, base_height, max_tilt_deg, min_base_height)
    upright = (~fallen).to(dtype=ee_pos.dtype)
    distance = torch.linalg.norm(ee_pos - object_pos, dim=-1)
    progress = (prev_distance - distance).clamp(-PROGRESS_CAP, PROGRESS_CAP)
    relative = torch.linalg.norm(object_vel - base_lin_vel_world, dim=-1)
    near = torch.exp(-distance / 0.12)
    caught = (distance < 0.09) & (relative < 0.25) & closed & ~fallen
    hold = caught.to(dtype=ee_pos.dtype)
    effort = -0.005 * actions.square().sum(dim=-1) * dt
    fall = -5.0 * fallen.to(dtype=ee_pos.dtype)
    if stage == "haul":
        error = (base_lin_vel[:, 0] - command[:, 0]).square()
        error = error + (base_lin_vel[:, 1] - command[:, 1]).square()
        error = error + (base_ang_vel[:, 2] - command[:, 2]).square()
        track = torch.exp(-error / 0.25)
        kept = grasped.to(dtype=ee_pos.dtype) * upright
        terms = {
            "hold": 2.0 * near * kept * dt,
            "track": 3.0 * track * kept * dt,
            "alive": kept * 0.3 * dt,
            "effort": effort,
            "fall": fall,
            "drop": -2.0 * (~grasped & ~fallen).to(dtype=ee_pos.dtype),
        }
        terms["tracking"] = track.detach()
    else:
        to_object = object_pos - ee_pos
        direction = to_object / distance.clamp(min=1e-4).unsqueeze(-1)
        incoming = -object_vel / torch.linalg.norm(object_vel, dim=-1, keepdim=True).clamp(min=1e-4)
        moving = torch.linalg.norm(object_vel, dim=-1) > 0.05
        align = (direction * incoming).sum(dim=-1).clamp(0.0, 1.0)
        align = torch.where(moving, align, torch.ones_like(align))
        terms = {
            "approach": 8.0 * progress * upright,
            "near": 1.5 * near * upright * dt,
            "align": 0.4 * align * upright * dt,
            "catch": 0.5 * hold,
            "alive": upright * 0.2 * dt,
            "effort": effort,
            "fall": fall,
        }
    scored = {name: value for name, value in terms.items() if name != "tracking"}
    reward = torch.stack(tuple(scored.values()), dim=0).sum(dim=0)
    return reward, terms, caught, fallen, distance.detach()


def ball_in_base(
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    level: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    seen_pos, seen_vel = obscure_motion(
        world_to_base(ball_pos, base_pos, base_quat),
        world_to_base(ball_vel, torch.zeros_like(base_pos), base_quat),
        level,
    )
    return seen_pos, seen_vel
