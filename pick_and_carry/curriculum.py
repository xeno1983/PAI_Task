"""Staged tasks on the Unitree Z1. Every stage is observation 81 and action 19.

Walk learns to stand, walk and turn on bumpy ground while the environment moves the arm
at random. Reach follows: the hand goes to a point in front of the body. Haul and carry
(the box task) and catch (a ball that rolls or flies) build on reach. Channels a stage does
not use stay zero.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pick_and_carry.arms import layouts
from pick_and_carry.config import LocomotionScales
from pick_and_carry.helpers import fallen_mask, world_to_base

STAGES = ("walk", "reach", "haul", "carry", "catch")
# Static, rolling slowly, rolling at random, thrown through the air, thrown with a noisy view.
CATCH_LEVELS = ("static", "slow", "random", "fly", "noisy")
FLYING_LEVELS = ("fly", "noisy")
LOCOMOTION_STAGES = ("walk",)
# The ball entity exists in these stages. In reach it is a ghost marker for the target point.
BALL_STAGES = ("reach", "catch")
# These stages drive the hand. Walk does not: the environment holds or moves the arm for it.
ARM_STAGES = ("reach", "haul", "carry", "catch")

WALK_COMMAND_SECONDS = 4.0
# Share of walk commands that ask the robot to stand still.
STAND_STILL_SHARE = 0.20
CATCH_HOLD_SECONDS = 1.0
BALL_RADIUS = 0.035
PROGRESS_CAP = 0.05
GRAVITY = 9.81
# The ball is predicted this far ahead. The prediction goes in the command slot, scaled down
# so a walking policy does not read it as a speed command.
PREDICT_SECONDS = 0.30
PREDICTION_SCALE = 0.10
BALL_VELOCITY_SCALE = 0.50

# Box in front of the body, in the base frame, where the hand is sent in reach and where
# a thrown ball is aimed. z is height above the ground.
REACH_X = (0.28, 0.50)
REACH_Y = (-0.20, 0.20)
REACH_Z = (0.30, 0.60)


SKILL_ORDER = ("walk", "reach", "haul", "carry")


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
    priors.append(("reach", "static"))
    priors.extend(_prior_skills("reach"))
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


def stage_log_dir(
    stage: str,
    catch_level: str = "static",
    arm: str = "z1",
    root: Path | None = None,
    algo: str = "ppo",
    *,
    easy: bool = False,
) -> Path:
    """Folder used by the launcher for this stage and learning algorithm.

    ``easy`` only applies to walk: flat ground, no arm wander / pushes, separate folder
    so a hard run is not overwritten.
    """
    if arm != "z1":
        raise ValueError(f"the arm is z1, got {arm}.")
    from pick_and_carry.algorithms import normalize_algo

    prefix = normalize_algo(algo)
    if stage == "carry":
        relative = f"logs/{prefix}_z1"
    elif stage == "walk":
        relative = f"logs/{prefix}_curriculum_walk_easy" if easy else f"logs/{prefix}_curriculum_walk"
    elif stage == "reach":
        relative = f"logs/{prefix}_curriculum_reach"
    elif stage == "haul":
        relative = f"logs/{prefix}_curriculum_haul"
    elif stage == "catch":
        if catch_level not in CATCH_LEVELS:
            raise ValueError(f"unknown catch level {catch_level}")
        relative = f"logs/{prefix}_curriculum_catch_{catch_level}"
    else:
        raise ValueError(f"unknown curriculum stage {stage}")
    base = Path(root) if root is not None else Path()
    return base / relative


def locomotion_action_mask(act_dim: int, device: torch.device | None = None) -> torch.Tensor:
    """True for leg dims only. Walk / haul hold the arm outside the policy."""
    from pick_and_carry.config import ACTION_LAYOUT, layout_slices

    mask = torch.zeros(act_dim, dtype=torch.bool, device=device)
    leg = layout_slices(ACTION_LAYOUT)["leg"]
    mask[leg] = True
    return mask


def holds_arm_pose(stage: str) -> bool:
    """Whether the policy leaves the arm alone. The environment holds it, or moves it at random in walk."""
    if stage not in STAGES:
        raise ValueError(f"unknown curriculum stage {stage}")
    return stage not in ARM_STAGES


def stage_layouts(stage: str, arm_dof: int) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Observation and action layout. Every stage uses the same one."""
    if stage not in STAGES:
        raise ValueError(f"unknown curriculum stage {stage}")
    return layouts(arm_dof)


def sample_command(count: int, device: torch.device | None = None, stand_share: float = 0.0) -> torch.Tensor:
    """Body-frame targets: forward m/s, left m/s, yaw rad/s.

    ``stand_share`` of the rows are zero, which asks the robot to stand still.
    """
    if device is None:
        device = torch.device("cpu")
    forward = torch.empty(count, device=device).uniform_(-0.25, 0.50)
    left = torch.empty(count, device=device).uniform_(-0.20, 0.20)
    yaw = torch.empty(count, device=device).uniform_(-0.60, 0.60)
    command = torch.stack((forward, left, yaw), dim=-1)
    if stand_share > 0.0:
        still = torch.rand(count, device=device) < stand_share
        command = torch.where(still[:, None], torch.zeros_like(command), command)
    return command


def sample_reach_target(base_pos: torch.Tensor) -> torch.Tensor:
    """World points in the reach box in front of a robot that starts facing +x."""
    count = base_pos.shape[0]
    device = base_pos.device
    target = base_pos.clone()
    target[:, 0] = base_pos[:, 0] + torch.empty(count, device=device).uniform_(*REACH_X)
    target[:, 1] = base_pos[:, 1] + torch.empty(count, device=device).uniform_(*REACH_Y)
    target[:, 2] = torch.empty(count, device=device).uniform_(*REACH_Z)
    return target


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
    elif level in FLYING_LEVELS:
        start_distance = torch.empty(count, device=device).uniform_(1.60, 2.40)
        angle = torch.empty(count, device=device).uniform_(-0.50, 0.50)
        position[:, 0] = base_pos[:, 0] + start_distance * torch.cos(angle)
        position[:, 1] = base_pos[:, 1] + start_distance * torch.sin(angle)
        position[:, 2] = torch.empty(count, device=device).uniform_(0.50, 0.90)
        aim = sample_reach_target(base_pos)
        flight = torch.empty(count, device=device).uniform_(0.80, 1.20)
        linear = (aim - position) / flight[:, None]
        linear[:, 2] = linear[:, 2] + 0.5 * GRAVITY * flight
    elif level == "random":
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
    if level in FLYING_LEVELS:
        angular = torch.zeros_like(angular)
    return position, linear, angular


def obscure_motion(position: torch.Tensor, velocity: torch.Tensor, level: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The noisy level shows a noisy position and speed instead of the exact ones."""
    if level != "noisy":
        return position, velocity
    return position + torch.randn_like(position) * 0.03, velocity + torch.randn_like(velocity) * 0.05


def feet_air_time_reward(
    air_time: torch.Tensor,
    last_contact: torch.Tensor,
    contact: torch.Tensor,
    command: torch.Tensor,
    dt: float,
    target: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """legged_gym ``feet_air_time``. Returns the reward, the new air time, and the new contact flags.

    A foot that touches down after ``target`` seconds in the air earns the excess. A shorter
    step costs the shortfall. Nothing is paid while the speed command is near zero.
    ``air_time``, ``last_contact`` and ``contact`` are per foot, shape (envs, 4).
    """
    touching = contact > 0.5
    filtered = touching | last_contact
    first_contact = (air_time > 0.0) & filtered
    air_time = air_time + dt
    reward = ((air_time - target) * first_contact.to(dtype=air_time.dtype)).sum(dim=-1)
    moving = command[:, :2].norm(dim=-1) > 0.1
    reward = reward * moving.to(dtype=reward.dtype)
    air_time = air_time * (~filtered).to(dtype=air_time.dtype)
    return reward, air_time, touching


MOVE_LIN = 0.10
MOVE_YAW = 0.20
STILL_SIGMA = 0.01


def walk_score(command: torch.Tensor, base_lin_vel: torch.Tensor, base_ang_vel: torch.Tensor) -> torch.Tensor:
    """0 to 1: how much of the commanded motion the body achieved. A log and the difficulty score.

    A moving command scores by the share of the commanded speed made along the command:
    1 at the commanded speed, 0 when standing still or at twice the speed or more, 0 when going
    the wrong way. Forward and yaw are scored separately and averaged when both are asked for.
    A near-zero command scores by staying still. Standing still on a moving command therefore
    scores about 0, unlike ``tracking``, which gives it more than half.
    """
    lin_command = command[:, :2]
    lin_sq = lin_command.square().sum(dim=-1)
    lin_ratio = (base_lin_vel[:, :2] * lin_command).sum(dim=-1) / lin_sq.clamp(min=1e-6)
    lin_score = (1.0 - (1.0 - lin_ratio).abs()).clamp(0.0, 1.0)
    yaw_command = command[:, 2]
    yaw_ratio = base_ang_vel[:, 2] * yaw_command / yaw_command.square().clamp(min=1e-6)
    yaw_score = (1.0 - (1.0 - yaw_ratio).abs()).clamp(0.0, 1.0)
    lin_on = (lin_sq.sqrt() > MOVE_LIN).to(dtype=lin_score.dtype)
    yaw_on = (yaw_command.abs() > MOVE_YAW).to(dtype=lin_score.dtype)
    asked = lin_on + yaw_on
    moving = (lin_on * lin_score + yaw_on * yaw_score) / asked.clamp(min=1.0)
    drift = base_lin_vel[:, :2].square().sum(dim=-1) + base_ang_vel[:, 2].square()
    still = torch.exp(-drift / STILL_SIGMA)
    return torch.where(asked > 0, moving, still).detach()


def locomotion_reward(
    *,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    command: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    leg_actions: torch.Tensor,
    prev_leg_actions: torch.Tensor,
    leg_vel: torch.Tensor,
    prev_leg_vel: torch.Tensor,
    leg_torque: torch.Tensor,
    foot_contact: torch.Tensor,
    air_time_reward: torch.Tensor,
    collisions: torch.Tensor,
    body_contact: torch.Tensor,
    max_tilt_deg: float,
    min_base_height: float,
    dt: float,
    scales: LocomotionScales | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Velocity tracking with the legged_gym penalties.

    The weights are in ``LocomotionScales``. The positive terms and penalties are summed and
    clipped at zero (``only_positive_rewards``), then the fall penalty is added. Only the
    legs count: the arm is held or driven by the stage, so its action never enters a penalty.
    ``tracking``, ``walk_score``, ``gait_score``, ``steady``, and ``jerk`` are logs and are not added to the reward.
    """
    scales = scales or LocomotionScales()
    fallen = fallen_mask(projected_gravity, base_height, max_tilt_deg, min_base_height) | body_contact
    sigma = scales.tracking_sigma
    lin_error = (command[:, :2] - base_lin_vel[:, :2]).square().sum(dim=-1)
    ang_error = (command[:, 2] - base_ang_vel[:, 2]).square()
    terms = {
        "tracking_lin_vel": scales.tracking_lin_vel * torch.exp(-lin_error / sigma) * dt,
        "tracking_ang_vel": scales.tracking_ang_vel * torch.exp(-ang_error / sigma) * dt,
        "lin_vel_z": scales.lin_vel_z * base_lin_vel[:, 2].square() * dt,
        "ang_vel_xy": scales.ang_vel_xy * base_ang_vel[:, :2].square().sum(dim=-1) * dt,
        "orientation": scales.orientation * projected_gravity[:, :2].square().sum(dim=-1) * dt,
        "base_height": scales.base_height * (base_height - scales.base_height_target).square() * dt,
        "torques": scales.torques * leg_torque.square().sum(dim=-1) * dt,
        "dof_acc": scales.dof_acc * ((prev_leg_vel - leg_vel) / dt).square().sum(dim=-1) * dt,
        "action_rate": scales.action_rate * (prev_leg_actions - leg_actions).square().sum(dim=-1) * dt,
        "feet_air_time": scales.feet_air_time * air_time_reward * dt,
        "collision": scales.collision * collisions * dt,
    }
    total = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    if scales.only_positive_rewards:
        total = total.clamp(min=0.0)
    terms["termination"] = scales.termination * fallen.to(dtype=total.dtype)
    reward = total + terms["termination"]

    error = lin_error + ang_error
    terms["tracking"] = torch.exp(-error / sigma).detach()
    terms["walk_score"] = walk_score(command, base_lin_vel, base_ang_vel)
    level = (-projected_gravity[:, 2]).clamp(0.0, 1.0)
    height = torch.exp(-(base_height - scales.base_height_target).square() / 0.008)
    terms["steady"] = (height * level).detach()
    contact = foot_contact.clamp(0.0, 1.0)
    front_left = contact[:, 0] * contact[:, 3]
    front_right = contact[:, 1] * contact[:, 2]
    rear_swing = (1.0 - contact[:, 0]) * (1.0 - contact[:, 3])
    front_swing = (1.0 - contact[:, 1]) * (1.0 - contact[:, 2])
    trot = front_left * front_swing + front_right * rear_swing
    moving = (command[:, :2].norm(dim=-1) > 0.1).to(dtype=trot.dtype)
    terms["gait_score"] = (trot * moving).detach()
    terms["jerk"] = (leg_actions - prev_leg_actions).abs().mean(dim=-1).detach()
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
    """Hold a carried box, or put the hand on a point or a ball (reach and catch).

    Returns reward, terms, caught, fallen, distance.
    """
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
        terms["walk_score"] = walk_score(command, base_lin_vel, base_ang_vel)
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
            # The legs hold still while the hand works.
            "still": -1.0 * (base_lin_vel[:, :2].square().sum(dim=-1) + base_ang_vel[:, 2].square()) * dt,
        }
    scored = {name: value for name, value in terms.items() if name not in ("tracking", "walk_score")}
    reward = torch.stack(tuple(scored.values()), dim=0).sum(dim=0)
    return reward, terms, caught, fallen, distance.detach()


def ball_in_base(
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    level: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ball position, velocity, and where it will be PREDICT_SECONDS from now, all in the base frame.

    The prediction follows a ballistic path while the ball is off the ground and a straight
    line while it rolls. The noisy level perturbs the position and velocity first.
    """
    seen_pos, seen_vel = obscure_motion(ball_pos, ball_vel, level)
    horizon = PREDICT_SECONDS
    future = seen_pos + seen_vel * horizon
    airborne = seen_pos[:, 2] > BALL_RADIUS + 0.03
    drop = 0.5 * GRAVITY * horizon * horizon
    future[:, 2] = torch.where(airborne, future[:, 2] - drop, torch.full_like(future[:, 2], BALL_RADIUS))
    future[:, 2] = future[:, 2].clamp(min=BALL_RADIUS)
    return (
        world_to_base(seen_pos, base_pos, base_quat),
        world_to_base(seen_vel, torch.zeros_like(base_pos), base_quat),
        world_to_base(future, base_pos, base_quat),
    )
