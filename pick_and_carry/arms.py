"""The Unitree Z1 on the Go2.

Six arm joints: observation 81, action CPG+arm (22). Every curriculum stage uses that layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pick_and_carry.config import ACTION_LAYOUT, OBS_LAYOUT, URDF_DIR, layout_dim

Z1_DOF = 6


def layouts(arm_dof: int) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """The one observation and action layout. Every stage uses it."""
    if arm_dof != Z1_DOF:
        raise ValueError(f"the Z1 arm has {Z1_DOF} joints, got {arm_dof}.")
    return OBS_LAYOUT, ACTION_LAYOUT


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
    # Grasp counts when the open amount is below this. The Z1 gripper is in radians.
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
        # Residual scale used when EnvConfig.arm_action_scale is not overridden.
        action_scale=0.12,
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


ARMS = {"z1": _z1()}


def get_arm(name: str) -> ArmSpec:
    try:
        return ARMS[name]
    except KeyError as exc:
        known = ", ".join(sorted(ARMS))
        raise ValueError(f"Unknown arm {name!r}. Choose one of: {known}.") from exc
