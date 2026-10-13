"""Staged tasks on the Unitree Z1. Every stage is observation 69 and action 19.

The previous stage initializes the next: stand, sit, walk, jump, rough, haul,
pick-and-place, then the catch levels. Channels a stage does not use stay zero.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pick_and_carry.arms import layouts
from pick_and_carry.helpers import fallen_mask, world_to_base

STAGES = ("carry", "walk", "jump", "stand", "sit", "rough", "haul", "catch")
POSTURE_STAGES = ("jump", "stand", "sit")
CATCH_LEVELS = ("static", "slow", "random", "noisy")

WALK_COMMAND_SECONDS = 4.0
CATCH_HOLD_SECONDS = 1.0
POSE_HOLD_SECONDS = 1.0
BALL_RADIUS = 0.035
PROGRESS_CAP = 0.05
STAND_HEIGHT = 0.40
SIT_HEIGHT = 0.28
JUMP_RISE = 0.08


SKILL_ORDER = ("stand", "sit", "walk", "jump", "rough", "haul", "carry")


def _prior_skills(stage: str) -> list[tuple[str, str]]:
    """Earlier stages in the curriculum, nearest first."""
    if stage not in SKILL_ORDER:
        return []
    index = SKILL_ORDER.index(stage)
    return [(name, "static") for name in reversed(SKILL_ORDER[:index])]


def earlier_stages(stage: str, catch_level: str = "static") -> list[tuple[str, str]]:
    """Same-sized policies that can initialize this stage, nearest previous first."""
    if stage != "catch":
        return _prior_skills(stage)
    if catch_level not in CATCH_LEVELS:
        raise ValueError(f"unknown catch level {catch_level}")
    index = CATCH_LEVELS.index(catch_level)
    priors = [("catch", level) for level in reversed(CATCH_LEVELS[:index])]
    priors.append(("carry", "static"))
    priors.extend(_prior_skills("carry"))
    return priors


def later_stages(stage: str, catch_level: str = "static") -> list[tuple[str, str]]:
    """Same-sized policies one step harder, nearest first. Used to practice an easier task again."""
    if stage == "catch":
        if catch_level not in CATCH_LEVELS:
            raise ValueError(f"unknown catch level {catch_level}")
        index = CATCH_LEVELS.index(catch_level)
        return [("catch", level) for level in CATCH_LEVELS[index + 1 :]]
    if stage not in SKILL_ORDER:
        return []
    index = SKILL_ORDER.index(stage)
    later = [(name, "static") for name in SKILL_ORDER[index + 1 :]]
    later.extend(("catch", level) for level in CATCH_LEVELS)
    return later


def retreat_choice(
    score: float | None,
    *,
    current: int,
    target: int,
    poor_streak: int,
    good_streak: int,
    patience: int = 8,
    poor: float = 0.30,
    good: float = 0.60,
) -> tuple[int, int, int]:
    """Move one step easier after repeated low scores, and back toward ``target`` after repeated high ones.

    Index 0 is the easiest setting. ``target`` is the hardest index this run is allowed to reach.
    A missing score leaves the streaks unchanged.
    """
    if score is None or target < 0:
        return current, poor_streak, good_streak
    current = min(max(current, 0), target)
    if score < poor:
        poor_streak += 1
        good_streak = 0
        if poor_streak >= patience and current > 0:
            return current - 1, 0, 0
        return current, poor_streak, good_streak
    if score > good:
        good_streak += 1
        poor_streak = 0
        if good_streak >= patience and current < target:
            return current + 1, 0, 0
        return current, poor_streak, good_streak
    return current, 0, 0


def stage_log_dir(stage: str, catch_level: str = "static", arm: str = "z1", root: Path | None = None) -> Path:
    """Folder used by the launcher for this stage."""
    if arm != "z1":
        raise ValueError(f"the arm is z1, got {arm}.")
    if stage == "carry":
        relative = "logs/ppo_z1"
    elif stage == "walk":
        relative = "logs/ppo_curriculum_walk"
    elif stage == "jump":
        relative = "logs/ppo_curriculum_jump"
    elif stage == "stand":
        relative = "logs/ppo_curriculum_stand"
    elif stage == "sit":
        relative = "logs/ppo_curriculum_sit"
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


def stage_layouts(stage: str, arm_dof: int) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Observation and action layout. Every stage uses the same one."""
    if stage not in STAGES:
        raise ValueError(f"unknown curriculum stage {stage}")
    return layouts(arm_dof)


def sample_command(count: int, device: torch.device | None = None) -> torch.Tensor:
    """Body-frame targets: forward m/s, left m/s, yaw rad/s."""
    if device is None:
        device = torch.device("cpu")
    forward = torch.empty(count, device=device).uniform_(-0.25, 0.50)
    left = torch.empty(count, device=device).uniform_(-0.20, 0.20)
    yaw = torch.empty(count, device=device).uniform_(-0.60, 0.60)
    return torch.stack((forward, left, yaw), dim=-1)


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


def posture_achieved(
    task: str,
    *,
    base_height: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    foot_contact: torch.Tensor,
    fallen: torch.Tensor,
    was_airborne: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whether this step holds the pose, and whether the feet are off the ground.

    Jump counts a landing only after a previous step left the ground. Stand and
    sit are a steady height with little horizontal speed.
    """
    speed = base_lin_vel[:, 0].square() + base_lin_vel[:, 1].square() + base_ang_vel[:, 2].square()
    contacts = foot_contact.sum(dim=-1)
    still = speed < 0.02
    upright = ~fallen
    airborne = upright & (contacts <= 0.5) & (base_height > STAND_HEIGHT + 0.03)
    if task == "jump":
        landed = was_airborne & upright & (contacts >= 2.0) & (base_height < STAND_HEIGHT + JUMP_RISE)
        return landed, airborne
    if task == "stand":
        held = upright & still & ((base_height - STAND_HEIGHT).abs() < 0.06)
        return held, airborne
    if task == "sit":
        held = upright & still & (contacts >= 3.0) & ((base_height - SIT_HEIGHT).abs() < 0.04)
        return held, airborne
    raise ValueError(f"unknown posture task {task}")


def pose_reward(
    task: str,
    *,
    base_height: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_lin_vel_world: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    foot_contact: torch.Tensor,
    actions: torch.Tensor,
    max_tilt_deg: float,
    min_base_height: float,
    dt: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Dense score for a jump, a stand, or a sit. ``tracking`` is the 0–1 pose score."""
    fallen = fallen_mask(projected_gravity, base_height, max_tilt_deg, min_base_height)
    upright = (~fallen).to(dtype=base_height.dtype)
    speed = base_lin_vel[:, 0].square() + base_lin_vel[:, 1].square() + base_ang_vel[:, 2].square()
    still = torch.exp(-speed / 0.08)
    contacts = foot_contact.clamp(0.0, 1.0).mean(dim=-1)
    if task == "stand":
        height = torch.exp(-(base_height - STAND_HEIGHT).square() / 0.008)
        pose = height * still
    elif task == "sit":
        height = torch.exp(-(base_height - SIT_HEIGHT).square() / 0.004)
        pose = height * still * contacts
    elif task == "jump":
        rise = (base_lin_vel_world[:, 2].clamp(min=0.0) / 1.2).clamp(max=1.0)
        clearance = ((base_height - STAND_HEIGHT) / JUMP_RISE).clamp(0.0, 1.0)
        pose = torch.maximum(rise, clearance)
    else:
        raise ValueError(f"unknown posture task {task}")
    terms = {
        "pose": 4.0 * pose * upright * dt,
        "alive": upright * 0.4 * dt,
        "effort": -0.01 * actions.square().sum(dim=-1) * dt,
        "fall": -5.0 * fallen.to(dtype=base_height.dtype),
    }
    reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    terms["tracking"] = pose.detach()
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
