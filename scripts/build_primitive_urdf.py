"""Write the mesh-free Go2 + Franka URDF used before official meshes are merged."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pick_and_carry.config import (  # noqa: E402
    ARM_LIMITS,
    DEFAULT_ARM_ANGLES,
    LEG_JOINTS,
    MOUNT_RPY,
    MOUNT_XYZ,
    PRIMITIVE_URDF,
    leg_limit,
)

LEGS = (
    ("FL", 0.1934, 0.0465, 0.0955),
    ("FR", 0.1934, -0.0465, -0.0955),
    ("RL", -0.1934, 0.0465, 0.0955),
    ("RR", -0.1934, -0.0465, -0.0955),
)

ARM_CHAIN = (
    ("panda_joint1", "panda_link0", "panda_link1", "0 0 0.333", "0 0 0", "0 0 1"),
    ("panda_joint2", "panda_link1", "panda_link2", "0 0 0", "-1.570796 0 0", "0 0 1"),
    ("panda_joint3", "panda_link2", "panda_link3", "0 -0.316 0", "1.570796 0 0", "0 0 1"),
    ("panda_joint4", "panda_link3", "panda_link4", "0.0825 0 0", "1.570796 0 0", "0 0 1"),
    ("panda_joint5", "panda_link4", "panda_link5", "-0.0825 0.384 0", "-1.570796 0 0", "0 0 1"),
    ("panda_joint6", "panda_link5", "panda_link6", "0 0 0", "1.570796 0 0", "0 0 1"),
    ("panda_joint7", "panda_link6", "panda_link7", "0.088 0 0", "1.570796 0 0", "0 0 1"),
)


def _inertial(mass: float, inertia: float | None = None) -> str:
    value = max(mass * 0.002, 1e-6) if inertia is None else inertia
    mass_text = f"{mass:.8g}"
    inertia_text = f"{value:.8g}"
    return (
        "    <inertial>\n"
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <mass value="{mass_text}"/>\n'
        f'      <inertia ixx="{inertia_text}" ixy="0" ixz="0" iyy="{inertia_text}" iyz="0" izz="{inertia_text}"/>\n'
        "    </inertial>"
    )


def _visual_collision(shape: str, rgba: str, material: str) -> str:
    return (
        "    <visual>\n"
        "      <geometry>\n"
        f"        {shape}\n"
        "      </geometry>\n"
        f'      <material name="{material}">\n'
        f'        <color rgba="{rgba}"/>\n'
        "      </material>\n"
        "    </visual>\n"
        "    <collision>\n"
        "      <geometry>\n"
        f"        {shape}\n"
        "      </geometry>\n"
        "    </collision>"
    )


def _revolute(name: str, parent: str, child: str, xyz: str, rpy: str, axis: str, lower: float, upper: float, effort: float) -> str:
    velocity = 30.1 if name in LEG_JOINTS else 2.175
    return f"""  <joint name="{name}" type="revolute">
    <origin xyz="{xyz}" rpy="{rpy}"/>
    <parent link="{parent}"/>
    <child link="{child}"/>
    <axis xyz="{axis}"/>
    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>
  </joint>
"""


def _leg(name: str, hip_x: float, hip_y: float, thigh_y: float) -> str:
    hip_lower, hip_upper, hip_effort = leg_limit(f"{name}_hip_joint")
    thigh_lower, thigh_upper, thigh_effort = leg_limit(f"{name}_thigh_joint")
    calf_lower, calf_upper, calf_effort = leg_limit(f"{name}_calf_joint")
    return f"""  <link name="{name}_hip">
{_inertial(0.678)}
{_visual_collision('<cylinder radius="0.04" length="0.04"/>', "0.25 0.25 0.28 1", f"{name}_hip_mat")}
  </link>
{_revolute(f"{name}_hip_joint", "base", f"{name}_hip", f"{hip_x} {hip_y} 0", "0 0 0", "1 0 0", hip_lower, hip_upper, hip_effort)}
  <link name="{name}_thigh">
{_inertial(1.152)}
{_visual_collision('<box size="0.04 0.03 0.21"/>', "0.35 0.35 0.38 1", f"{name}_thigh_mat")}
  </link>
{_revolute(f"{name}_thigh_joint", f"{name}_hip", f"{name}_thigh", f"0 {thigh_y} 0", "0 0 0", "0 1 0", thigh_lower, thigh_upper, thigh_effort)}
  <link name="{name}_calf">
{_inertial(0.154)}
{_visual_collision('<cylinder radius="0.015" length="0.20"/>', "0.2 0.2 0.22 1", f"{name}_calf_mat")}
  </link>
{_revolute(f"{name}_calf_joint", f"{name}_thigh", f"{name}_calf", "0 0 -0.213", "0 0 0", "0 1 0", calf_lower, calf_upper, calf_effort)}
  <link name="{name}_foot">
{_inertial(0.04, 9.6e-6)}
{_visual_collision('<sphere radius="0.022"/>', "0.05 0.05 0.05 1", f"{name}_foot_mat")}
  </link>
  <joint name="{name}_foot_joint" type="fixed">
    <origin xyz="0 0 -0.213" rpy="0 0 0"/>
    <parent link="{name}_calf"/>
    <child link="{name}_foot"/>
  </joint>
"""


def _arm() -> str:
    chunks = [
        f"""  <link name="panda_link0">
{_inertial(0.4)}
{_visual_collision('<cylinder radius="0.06" length="0.05"/>', "0.9 0.9 0.92 1", "panda_link0_mat")}
  </link>
"""
    ]
    for name, parent, child, xyz, rpy, axis in ARM_CHAIN:
        lower, upper, effort = ARM_LIMITS[name]
        chunks.append(
            f"""  <link name="{child}">
{_inertial(0.35)}
{_visual_collision('<cylinder radius="0.04" length="0.12"/>', "0.95 0.95 0.97 1", f"{child}_mat")}
  </link>
{_revolute(name, parent, child, xyz, rpy, axis, lower, upper, effort)}"""
        )
    chunks.append(
        f"""  <link name="panda_link8">
{_inertial(0.15)}
{_visual_collision('<cylinder radius="0.035" length="0.04"/>', "0.85 0.85 0.88 1", "panda_link8_mat")}
  </link>
  <joint name="panda_joint8" type="fixed">
    <origin xyz="0 0 0.107" rpy="0 0 0"/>
    <parent link="panda_link7"/>
    <child link="panda_link8"/>
  </joint>
  <link name="panda_hand">
{_inertial(0.25)}
{_visual_collision('<box size="0.04 0.08 0.06"/>', "0.15 0.15 0.15 1", "panda_hand_mat")}
  </link>
  <joint name="panda_hand_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 -0.785398"/>
    <parent link="panda_link8"/>
    <child link="panda_hand"/>
  </joint>
  <link name="panda_leftfinger">
{_inertial(0.05)}
{_visual_collision('<box size="0.01 0.012 0.04"/>', "0.7 0.7 0.72 1", "panda_leftfinger_mat")}
  </link>
  <joint name="panda_finger_joint1" type="prismatic">
    <origin xyz="0 0 0.0584" rpy="0 0 0"/>
    <parent link="panda_hand"/>
    <child link="panda_leftfinger"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.04" effort="20" velocity="0.2"/>
  </joint>
  <link name="panda_rightfinger">
{_inertial(0.05)}
{_visual_collision('<box size="0.01 0.012 0.04"/>', "0.7 0.7 0.72 1", "panda_rightfinger_mat")}
  </link>
  <joint name="panda_finger_joint2" type="prismatic">
    <origin xyz="0 0 0.0584" rpy="0 0 0"/>
    <parent link="panda_hand"/>
    <child link="panda_rightfinger"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.04" effort="20" velocity="0.2"/>
  </joint>
  <link name="panda_hand_tcp">
{_inertial(0.001, 1e-6)}
    <visual>
      <geometry><sphere radius="0.008"/></geometry>
      <material name="panda_hand_tcp_mat"><color rgba="1 0.4 0.1 1"/></material>
    </visual>
  </link>
  <joint name="panda_hand_tcp_joint" type="fixed">
    <origin xyz="0 0 0.1034" rpy="0 0 0"/>
    <parent link="panda_hand"/>
    <child link="panda_hand_tcp"/>
  </joint>
"""
    )
    return "".join(chunks)


def build_urdf() -> str:
    xyz = f"{MOUNT_XYZ[0]:.6f} {MOUNT_XYZ[1]:.6f} {MOUNT_XYZ[2]:.6f}"
    rpy = f"{MOUNT_RPY[0]:.6f} {MOUNT_RPY[1]:.6f} {MOUNT_RPY[2]:.6f}"
    legs = "".join(_leg(*spec) for spec in LEGS)
    arm_note = ", ".join(f"{name}={angle}" for name, angle in DEFAULT_ARM_ANGLES.items())
    return f"""<?xml version="1.0"?>
<robot name="quadruped_arm">
  <!--
    Mesh-free Go2 + Franka model for Colab / Genesis.
    arm_mount_joint welds panda_link0 onto the Go2 base.
    Frame: Go2 +X is forward, +Z is up. Mount xyz={MOUNT_XYZ} rpy={MOUNT_RPY}
    sits slightly behind the base origin and above the chassis.
    Arm link masses are reduced (about 3 kg total) so the dog can stand.
    Replace this file's kinematics with scripts/merge_urdf.py once go2.urdf
    and panda.urdf are available. Suggested reset arm pose: {arm_note}.
  -->
  <link name="base">
    <inertial>
      <origin xyz="0.021112 0 -0.005366" rpy="0 0 0"/>
      <mass value="6.921"/>
      <inertia ixx="0.02448" ixy="0.00012166" ixz="0.0014849" iyy="0.098077" iyz="-0.0000312" izz="0.107"/>
    </inertial>
    <visual>
      <geometry><box size="0.376 0.12 0.114"/></geometry>
      <material name="base_mat"><color rgba="0.15 0.15 0.17 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.376 0.12 0.114"/></geometry>
    </collision>
  </link>
{legs}
  <joint name="arm_mount_joint" type="fixed">
    <origin xyz="{xyz}" rpy="{rpy}"/>
    <parent link="base"/>
    <child link="panda_link0"/>
  </joint>
{_arm()}
</robot>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write assets/urdf/quadruped_arm.urdf")
    parser.add_argument("--output", type=Path, default=PRIMITIVE_URDF)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_urdf(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
