"""Shared specification for the Go2 + Unitree Z1 pick-and-carry task.

The mount pose is the single source of truth for ``scripts/build_z1_urdf.py``.
Go2 ``base`` uses +X forward and +Z up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pick_and_carry.terrain import GOAL_X_RANGE, OBJECT_X_RANGE

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
URDF_DIR = PACKAGE_ROOT / "assets" / "urdf"

# Slightly aft of the base origin, just above the chassis top (~0.057 m).
MOUNT_XYZ = (-0.05, 0.0, 0.08)
MOUNT_RPY = (0.0, 0.0, 0.0)

LEG_JOINTS = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
FOOT_LINKS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

# Per-leg order matches LEG_JOINTS. Angles are radians, in the standing pose
# used by Unitree's Go2 locomotion examples.
DEFAULT_LEG_ANGLES = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
}
LEG_LIMITS = {
    "hip": (-1.0472, 1.0472, 23.7),
    "thigh": (-1.5708, 3.4907, 23.7),
    "calf": (-2.7227, -0.83776, 45.43),
}

# (name, dim). Positions of the object, goal, and end effector are expressed
# in the Go2 base frame.
OBS_LAYOUT = (
    ("leg_pos", 12),
    ("leg_vel", 12),
    ("projected_gravity", 3),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("base_height", 1),
    ("foot_contact", 4),
    ("arm_pos", 6),
    ("arm_vel", 6),
    ("gripper_pos", 1),
    ("gripper_vel", 1),
    ("ee_pos", 3),
    ("object_pos", 3),
    ("goal_pos", 3),
    ("grasped", 1),
    ("phase", 4),
    ("command", 3),
    # The leg action applied on the previous step, as in legged_gym and Walk These Ways.
    ("last_leg_action", 12),
)
# Policy actions: CPG params drive the 12 legs; arm/gripper are residual Δq / Δg
# on top of IK / phase baselines (see ``pick_and_carry.ik`` and ``control``).
# CPG 15 = classic 10 + dphi_ab/a/b + couple + entrain (uneven-terrain phase adapt).
# Old 10-D / act-17 checkpoints are incompatible.
CPG_PARAM_DIM = 15
ACTION_LAYOUT = (
    ("cpg", CPG_PARAM_DIM),
    ("arm", 6),
    ("gripper", 1),
)


def layout_dim(layout: tuple[tuple[str, int], ...]) -> int:
    return sum(dim for _, dim in layout)


def layout_slices(layout: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    cursor = 0
    slices: dict[str, slice] = {}
    for name, dim in layout:
        slices[name] = slice(cursor, cursor + dim)
        cursor += dim
    return slices


NUM_OBS = layout_dim(OBS_LAYOUT)
NUM_ACTIONS = layout_dim(ACTION_LAYOUT)


def leg_limit(joint_name: str) -> tuple[float, float, float]:
    if joint_name.endswith("_hip_joint"):
        return LEG_LIMITS["hip"]
    if joint_name.endswith("_thigh_joint"):
        return LEG_LIMITS["thigh"]
    if joint_name.endswith("_calf_joint"):
        return LEG_LIMITS["calf"]
    raise KeyError(joint_name)


@dataclass
class RewardScales:
    approach_ee: float = 1.5
    approach_base: float = 0.4
    grasp_shape: float = 6.0
    grasp_bonus: float = 5.0
    transport: float = 2.0
    place_bonus: float = 10.0
    fall: float = 5.0
    energy: float = 0.02
    alive: float = 0.3
    # Reward per metre of closing the gap. The exponential approach term is
    # nearly flat while the hand is still half a metre away.
    progress_ee: float = 8.0
    progress_goal: float = 8.0
    progress_base: float = 8.0
    forward: float = 1.5
    release: float = 4.0


@dataclass
class LocomotionScales:
    """Walking reward weights. Defaults follow legged_gym, with extra walk incentives.

    Every term is multiplied by ``dt``. Negative weights are penalties.
    When ``only_positive_rewards`` is set, *positive-scale* base terms are summed and
    clamped at zero; *penalty* terms always apply (so stepping costs cannot erase
    tracking by dragging the whole sum through clamp). walk/gait/standstill/termination
    are added after that.
    """

    tracking_lin_vel: float = 2.5
    tracking_ang_vel: float = 0.5
    tracking_sigma: float = 0.25
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = 0.0
    base_height: float = -20.0
    base_height_target: float = 0.30
    torques: float = -1e-5
    dof_acc: float = -2.5e-7
    # Milder than -0.01 so CPG stepping is not punished into a freeze/shuffle.
    action_rate: float = -0.002
    feet_air_time: float = 1.5
    # Seconds in the air that earn a positive step. legged_gym uses 0.5.
    feet_air_time_target: float = 0.5
    # Per thigh or calf touching something.
    collision: float = -1.0
    # Charged once on a fall, after base shaping.
    termination: float = -5.0
    only_positive_rewards: bool = True
    # Fraction of commanded motion achieved (0 when standing on a move command).
    walk_score: float = 1.5
    # Diagonal / anti-phase contact while a linear command is active (post base clamp).
    gait: float = 1.5
    # Soft cost for (1 - walk_score) when cmd asks to move (post base clamp).
    standstill: float = -2.0
    # Flat cost when cmd asks to move but body speed is below the threshold (post base clamp).
    standstill_vel: float = -1.0
    standstill_vel_threshold: float = 0.08


@dataclass
class EnvConfig:
    num_envs: int = 64
    backend: str = "gpu"
    show_viewer: bool = False
    # launcher records this file and opens it instead.
    record_path: str | None = None
    # Add a camera without starting a file, so a script can render frames itself.
    enable_camera: bool = False
    # One still of the first robots, written at the start of training. None skips it.
    fleet_snapshot: str | None = None
    camera_res: tuple[int, int] = (960, 540)
    # Side view of the whole course, from the start pad through the goal pad.
    camera_pos: tuple[float, float, float] = (2.0, -4.6, 2.6)
    camera_lookat: tuple[float, float, float] = (2.0, 0.0, 0.15)
    camera_fov: float = 58.0
    # Bumpy strips between flat pads. Pickup and release stay on the pads.
    terrain: bool = True
    terrain_amplitude: float = 0.035
    terrain_horizontal_scale: float = 0.05
    # Flat treads on the travel strips. Pads under the box and the goal stay flat.
    terrain_steps: bool = False
    urdf_path: str | None = None
    # split: legs while walking, arm while picking up or releasing.
    # walk: legs the whole time, arm held still, reward is travel and not falling.
    # both: legs and arm together for the whole episode.
    control_split: str = "split"
    # The arm is the Unitree Z1 (6 DoF, about 5.2 kg). Observation 81, action CPG+arm (22).
    arm: str = "z1"
    # Wrist camera on the hand. Grasp uses the segmented cube, not the simulator position.
    wrist_camera: bool = False
    wrist_camera_res: tuple[int, int] = (64, 48)
    wrist_camera_fov: float = 70.0
    # 0 keeps the single pick-and-carry course. Above 0 builds a wide repeating yard.
    yard_stations: int = 0
    # Each episode draws a new bump pattern. A yard recording keeps one pattern.
    randomize_terrain: bool = True
    # After a place, move on to the next supply instead of ending the episode.
    cycle_supplies: bool = False
    dt: float = 0.02
    substeps: int = 2
    episode_length_s: float = 20.0
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.42)
    base_init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    env_spacing: tuple[float, float] = (3.0, 3.0)
    # Actions are Gaussian samples with no squashing, so the clip is only a guard.
    action_clip: float = 100.0
    # Radians of leg motion per action unit. The Genesis Go2 example and legged_gym use 0.25.
    leg_action_scale: float = 0.25
    # Arm policy is a residual on IK (radians per action unit). Kept small so
    # RL fine-tunes contact / tracking rather than inventing full reaches.
    arm_action_scale: float = 0.12
    # Gripper residual scale (joint radians per action unit) around the phase baseline.
    gripper_residual_scale: float = 0.35
    # The Genesis Go2 example uses 20 and 0.5 for the bare Go2. With the Z1 on its back the
    # robot is about 5 kg heavier and 20 cannot hold it: it sinks to the joint limits and
    # bounces. Holding the standing pose, 30 sags to 0.23 m, 40 to 0.29 m, 60 to 0.305 m,
    # and 80 to 0.313 m. 60 and 1.5 keep the damping ratio of the Go2 value with margin
    # for the box in haul and carry.
    leg_kp: float = 60.0
    leg_kd: float = 1.5
    arm_kp: float = 120.0
    arm_kd: float = 12.0
    gripper_kp: float = 40.0
    gripper_kd: float = 4.0
    gripper_open: float = 0.04
    gripper_closed: float = 0.012
    grasp_distance: float = 0.12
    cube_size: float = 0.04
    cube_density: float = 800.0
    # On the flat pickup pad, beyond the first bumpy strip.
    object_x_range: tuple[float, float] = OBJECT_X_RANGE
    object_y_range: tuple[float, float] = (-0.15, 0.15)
    # On the flat goal pad, beyond the second bumpy strip.
    goal_x_range: tuple[float, float] = GOAL_X_RANGE
    goal_y_range: tuple[float, float] = (-0.25, 0.25)
    # Inference clicks. None keeps the random pad sample.
    fixed_object_xy: tuple[float, float] | None = None
    fixed_goal_xy: tuple[float, float] | None = None
    # Base-to-object distance that starts the pickup. Object-to-goal distance
    # that starts the release while the cube is held.
    pickup_distance: float = 0.55
    release_distance: float = 0.45
    goal_radius: float = 0.28
    place_height: float = 0.08
    goal_marker_size: tuple[float, float, float] = (0.30, 0.30, 0.01)
    min_base_height: float = 0.18
    max_tilt_deg: float = 60.0
    contact_force_threshold: float = 1.0
    approach_sigma: float = 0.25
    transport_sigma: float = 0.40
    base_sigma: float = 0.60
    reward_scales: RewardScales = field(default_factory=RewardScales)
    locomotion_scales: LocomotionScales = field(default_factory=LocomotionScales)
    # Training-only robustness for walking. Playback leaves both off.
    push_robots: bool = False
    push_interval_s: float = 15.0
    push_max_vel: float = 0.5
    randomize_friction: bool = False
    # Walk only (opt-in): randomly wander the held arm. Default off — policy does not drive the arm.
    arm_disturb: bool = False
    # Fraction of each joint's range from the reset pose that a random target may use.
    arm_disturb_scale: float = 0.5
    arm_disturb_interval_s: float = 2.0
    # Per-step share of the way the arm moves toward its random target.
    arm_disturb_rate: float = 0.06
    friction_range: tuple[float, float] = (0.5, 1.25)
    max_collision_pairs: int = 512
    seed: int = 0
    # Every stage shares the Z1 observation. carry is pick-and-place.
    curriculum: str = "carry"
    catch_level: str = "static"
    catch_hold_steps: int = 50

    @property
    def max_episode_length(self) -> int:
        return int(round(self.episode_length_s / self.dt))
