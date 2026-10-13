"""Shared specification for the Go2 + Franka pick-and-carry task.

The mount pose is the single source of truth for both the primitive URDF and
the official-URDF merger. Go2 ``base`` uses +X forward and +Z up. Franka
``panda_link0`` also uses +Z up, so a zero rpy keeps the arm upright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pick_and_carry.terrain import GOAL_X_RANGE, OBJECT_X_RANGE

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
URDF_DIR = PACKAGE_ROOT / "assets" / "urdf"
PRIMITIVE_URDF = URDF_DIR / "quadruped_arm.urdf"
MERGED_URDF = URDF_DIR / "quadruped_arm_merged.urdf"

# Slightly aft of the base origin, just above the chassis top (~0.057 m).
MOUNT_XYZ = (-0.05, 0.0, 0.08)
MOUNT_RPY = (0.0, 0.0, 0.0)
MOUNT_JOINT = "arm_mount_joint"
BASE_LINK_CANDIDATES = ("base", "base_link", "trunk")
ARM_ROOT_LINK = "panda_link0"

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
ARM_JOINTS = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
FOOT_LINKS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
EE_LINK_CANDIDATES = ("panda_hand_tcp", "panda_grasptarget", "panda_hand", "panda_link8")

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
# Forward and partly lowered, so action 0 starts in the workspace above the
# cube. Joint 2 at -0.785 points the hand straight up, which the old ±0.4 rad
# action range could never leave. Joint 4 stays inside [-3.07, -0.07].
DEFAULT_ARM_ANGLES = {
    "panda_joint1": 0.0,
    "panda_joint2": 0.9,
    "panda_joint3": 0.0,
    "panda_joint4": -1.4,
    "panda_joint5": 0.0,
    "panda_joint6": 1.8,
    "panda_joint7": 0.7,
}

LEG_LIMITS = {
    "hip": (-1.0472, 1.0472, 23.7),
    "thigh": (-1.5708, 3.4907, 23.7),
    "calf": (-2.7227, -0.83776, 45.43),
}
ARM_LIMITS = {
    "panda_joint1": (-2.8973, 2.8973, 87.0),
    "panda_joint2": (-1.7628, 1.7628, 87.0),
    "panda_joint3": (-2.8973, 2.8973, 87.0),
    "panda_joint4": (-3.0718, -0.0698, 87.0),
    "panda_joint5": (-2.8973, 2.8973, 12.0),
    "panda_joint6": (-0.0175, 3.7525, 12.0),
    "panda_joint7": (-2.8973, 2.8973, 12.0),
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
)
# [12 leg positions | 6 Z1 arm positions | 1 gripper open/close]
ACTION_LAYOUT = (
    ("leg", 12),
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


def default_urdf_path() -> Path:
    """Prefer a merged mesh URDF, then official visuals on the primitive body."""
    if MERGED_URDF.is_file():
        return MERGED_URDF
    from pick_and_carry.visual_urdf import styled_urdf_path

    styled = styled_urdf_path()
    if styled is not None:
        return styled
    return PRIMITIVE_URDF


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
class EnvConfig:
    num_envs: int = 64
    backend: str = "gpu"
    show_viewer: bool = False
    # mp4 path. The interactive viewer crashes on this machine, so the desktop
    # launcher records this file and opens it instead.
    record_path: str | None = None
    # Add a camera without starting a file, so a script can render frames itself.
    enable_camera: bool = False
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
    control_mode: str = "position"  # "position" or "torque"
    # Kinematic attach once the gripper is closed near the cube. This is the
    # curriculum stage; set False to require a physical finger grasp.
    assist_grasp: bool = True
    # Hold the standing pose. The policy does not command the legs.
    freeze_legs: bool = False
    # False: while walking, the policy commands the legs around the standing pose.
    # True: a fixed diagonal trot walks, and the policy only adds a small residual.
    scripted_gait: bool = False
    # split: legs while walking, arm while picking up or releasing.
    # walk: legs the whole time, arm held still, reward is travel and not falling.
    # both: legs and arm together for the whole episode.
    control_split: str = "split"
    # The arm is the Unitree Z1 (6 DoF, about 5.2 kg). Observation 69, action 19.
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
    action_clip: float = 1.0
    # Radians of leg motion at a full action. Enough for a step around the standing pose.
    leg_action_scale: float = 0.50
    # One action unit. Large enough that joint 2 can go from the forward pose
    # (0.9) to the low reach (about 1.6) and back.
    arm_action_scale: float = 1.2
    leg_kp: float = 80.0
    leg_kd: float = 4.0
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
    gait_period: float = 0.40
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
    max_collision_pairs: int = 512
    seed: int = 0
    # Every stage shares the Z1 observation. carry is pick-and-place.
    curriculum: str = "carry"
    catch_level: str = "static"
    catch_hold_steps: int = 50

    @property
    def max_episode_length(self) -> int:
        return int(round(self.episode_length_s / self.dt))
