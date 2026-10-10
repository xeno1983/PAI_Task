"""Selectable arms on the Go2.

Panda is the default: 7 arm joints, observation 68, action 20. Existing
checkpoints load only with that arm. The Unitree Z1 has 6 arm joints, so its
observation is 66 and its action is 19. Those runs are stored separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pick_and_carry.config import (
    ARM_JOINTS,
    ARM_LIMITS,
    DEFAULT_ARM_ANGLES,
    FINGER_JOINTS,
    URDF_DIR,
    layout_dim,
)

PANDA_DOF = 7
Z1_DOF = 6


def layouts(arm_dof: int) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Observation and action layouts. Only the arm block changes with the manipulator."""
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
        ("goal_pos", 3),
        ("grasped", 1),
        ("phase", 4),
    )
    action = (("leg", 12), ("arm", arm_dof), ("gripper", 1))
    return observation, action


@dataclass(frozen=True)
class ArmSpec:
    name: str
    joints: tuple[str, ...]
    fingers: tuple[str, ...]
    limits: dict[str, tuple[float, float, float]]
    default_angles: dict[str, float]
    ee_links: tuple[str, ...]
    tcp_link: str
    # Joint position at action +1 (open) and action -1 (closed).
    joint_open: float
    joint_closed: float
    # Grasp counts when the open amount is below this. Panda is metres, Z1 is radians.
    closed_threshold: float
    action_scale: float
    kp: float
    kd: float
    gripper_kp: float
    gripper_kd: float
    urdf_name: str
    # Camera pose in the tcp frame. OpenGL looks along -Z.
    wrist_rotation: tuple[tuple[float, float, float], ...]
    wrist_translation: tuple[float, float, float]
    log_dir_name: str

    @property
    def dof(self) -> int:
        return len(self.joints)

    @property
    def num_obs(self) -> int:
        return layout_dim(layouts(self.dof)[0])

    @property
    def num_actions(self) -> int:
        return layout_dim(layouts(self.dof)[1])

    @property
    def urdf_path(self):
        return URDF_DIR / self.urdf_name

    def wrist_offset(self) -> np.ndarray:
        offset = np.eye(4, dtype=np.float32)
        offset[:3, :3] = np.asarray(self.wrist_rotation, dtype=np.float32)
        offset[:3, 3] = np.asarray(self.wrist_translation, dtype=np.float32)
        return offset


def _panda() -> ArmSpec:
    limits = {name: ARM_LIMITS[name] for name in ARM_JOINTS}
    limits["panda_finger_joint1"] = (0.0, 0.04, 20.0)
    limits["panda_finger_joint2"] = (0.0, 0.04, 20.0)
    return ArmSpec(
        name="panda",
        joints=ARM_JOINTS,
        fingers=FINGER_JOINTS,
        limits=limits,
        default_angles=dict(DEFAULT_ARM_ANGLES),
        ee_links=("panda_hand_tcp", "panda_grasptarget", "panda_hand", "panda_link8"),
        tcp_link="panda_hand_tcp",
        joint_open=0.04,
        joint_closed=0.0,
        closed_threshold=0.012,
        action_scale=1.2,
        kp=120.0,
        kd=12.0,
        gripper_kp=40.0,
        gripper_kd=4.0,
        urdf_name="quadruped_arm.urdf",
        wrist_rotation=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        wrist_translation=(0.0, 0.0, 0.045 - 0.1034),
        log_dir_name="ppo_walk",
    )


def _z1() -> ArmSpec:
    """Unitree Z1. Joint ranges and the 5.2 kg mass budget come from the official description."""
    deg = math.pi / 180.0
    limits = {
        "z1_joint1": (-150.0 * deg, 150.0 * deg, 30.0),
        "z1_joint2": (0.0, 170.0 * deg, 60.0),
        "z1_joint3": (-165.0 * deg, 0.0, 30.0),
        "z1_joint4": (-87.0 * deg, 87.0 * deg, 30.0),
        "z1_joint5": (-77.0 * deg, 77.0 * deg, 30.0),
        "z1_joint6": (-160.0 * deg, 160.0 * deg, 30.0),
        "z1_jointGripper": (-90.0 * deg, 0.0, 30.0),
    }
    # Shoulder and elbow put the gripper about 0.39 m in front of the base, 0.25 m up.
    defaults = {
        "z1_joint1": 0.0,
        "z1_joint2": 1.76,
        "z1_joint3": -1.00,
        "z1_joint4": 0.0,
        "z1_joint5": 0.0,
        "z1_joint6": 0.0,
    }
    return ArmSpec(
        name="z1",
        joints=tuple(defaults.keys()),
        fingers=("z1_jointGripper",),
        limits=limits,
        default_angles=defaults,
        ee_links=("z1_tcp",),
        tcp_link="z1_tcp",
        joint_open=-0.5 * math.pi,
        joint_closed=0.0,
        closed_threshold=0.20,
        action_scale=1.0,
        kp=80.0,
        kd=8.0,
        gripper_kp=20.0,
        gripper_kd=2.0,
        urdf_name="quadruped_z1.urdf",
        # Tool +X is forward. Point the camera's -Z along that axis.
        wrist_rotation=((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
        wrist_translation=(-0.03, 0.0, 0.0),
        log_dir_name="ppo_z1",
    )


ARMS = {"panda": _panda(), "z1": _z1()}


def get_arm(name: str) -> ArmSpec:
    try:
        return ARMS[name]
    except KeyError as exc:
        known = ", ".join(sorted(ARMS))
        raise ValueError(f"Unknown arm {name!r}. Choose one of: {known}.") from exc
