"""Write a Go2 + Unitree Z1 URDF with primitive shapes.

Joint origins, masses, inertia, and limits follow Unitree's z1_description
(BSD-3-Clause). Meshes are replaced by cylinders so the file stays self-contained.
The arm, including the gripper, is about 5.2 kg.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pick_and_carry.arms import get_arm  # noqa: E402
from pick_and_carry.config import MOUNT_XYZ, URDF_DIR  # noqa: E402

# name, mass, com xyz, inertia ixx ixy ixz iyy iyz izz
LINKS = (
    ("z1_link00", 0.47247481, (-0.00334984, -0.00013615, 0.02495843), (0.00037937, -3.5e-7, -1.037e-5, 0.00041521, -9.9e-7, 0.00053066)),
    ("z1_link01", 0.67332551, (0.00000247, -0.00025198, 0.02317169), (0.00128328, -6e-8, -4e-7, 0.00071931, 5e-7, 0.00083936)),
    ("z1_link02", 1.19132258, (-0.11012601, 0.00240029, 0.00158266), (0.00102138, 0.00062358, 5.13e-6, 0.02429457, -2.1e-6, 0.02466114)),
    ("z1_link03", 0.83940874, (0.10609208, -0.00541815, 0.03476383), (0.00108061, -8.669e-5, -0.00208102, 0.00954238, -1.332e-5, 0.00886621)),
    ("z1_link04", 0.56404563, (0.04366681, 0.00364738, -0.00170192), (0.00031576, 8.13e-5, 4.091e-5, 0.00092996, -5.96e-6, 0.00097912)),
    ("z1_link05", 0.38938492, (0.03121533, 0.0, 0.00646316), (0.00017605, 4e-7, 5.689e-5, 0.00055896, -1.3e-7, 0.00053860)),
    ("z1_link06", 0.28875807, (0.02415690, -0.00017355, -0.00143876), (0.00018328, 1.22e-6, 5.4e-7, 0.00014750, 8e-8, 0.00014680)),
    ("z1_gripper_stator", 0.52603655, (0.04764427, -0.00035819, -0.00249162), (0.00038683, -3.59e-6, 7.662e-5, 0.00068614, 2.09e-6, 0.00066293)),
    ("z1_gripper_mover", 0.27621302, (0.01320633, 0.00476708, 0.00380534), (0.00017716, 1.683e-5, -1.786e-5, 0.00026787, 2.62e-6, 0.00035728)),
)

# name, parent, child, xyz, axis, (lower, upper, effort)
JOINTS = (
    ("z1_joint1", "z1_link00", "z1_link01", "0 0 0.0585", "0 0 1"),
    ("z1_joint2", "z1_link01", "z1_link02", "0 0 0.045", "0 1 0"),
    ("z1_joint3", "z1_link02", "z1_link03", "-0.35 0 0", "0 1 0"),
    ("z1_joint4", "z1_link03", "z1_link04", "0.218 0 0.057", "0 1 0"),
    ("z1_joint5", "z1_link04", "z1_link05", "0.07 0 0", "0 0 1"),
    ("z1_joint6", "z1_link05", "z1_link06", "0.0492 0 0", "1 0 0"),
)


def _inertial(mass: float, com: tuple[float, float, float], inertia: tuple[float, ...]) -> ET.Element:
    node = ET.Element("inertial")
    origin = ET.SubElement(node, "origin")
    origin.set("xyz", f"{com[0]:.8g} {com[1]:.8g} {com[2]:.8g}")
    origin.set("rpy", "0 0 0")
    ET.SubElement(node, "mass").set("value", f"{mass:.8g}")
    ixx, ixy, ixz, iyy, iyz, izz = inertia
    tensor = ET.SubElement(node, "inertia")
    tensor.set("ixx", f"{ixx:.8g}")
    tensor.set("ixy", f"{ixy:.8g}")
    tensor.set("ixz", f"{ixz:.8g}")
    tensor.set("iyy", f"{iyy:.8g}")
    tensor.set("iyz", f"{iyz:.8g}")
    tensor.set("izz", f"{izz:.8g}")
    return node


def _cylinder(parent: ET.Element, radius: float, length: float, xyz: str, rpy: str) -> None:
    for tag in ("visual", "collision"):
        node = ET.SubElement(parent, tag)
        origin = ET.SubElement(node, "origin")
        origin.set("xyz", xyz)
        origin.set("rpy", rpy)
        geom = ET.SubElement(node, "geometry")
        cyl = ET.SubElement(geom, "cylinder")
        cyl.set("radius", f"{radius:.4g}")
        cyl.set("length", f"{length:.4g}")


def _box(parent: ET.Element, size: str) -> None:
    for tag in ("visual", "collision"):
        node = ET.SubElement(parent, tag)
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "box").set("size", size)


def _revolute(name: str, parent: str, child: str, xyz: str, axis: str, lower: float, upper: float, effort: float) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": "revolute"})
    ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": axis})
    ET.SubElement(joint, "limit", {"lower": f"{lower:.6g}", "upper": f"{upper:.6g}", "effort": f"{effort:.4g}", "velocity": "3.1415"})
    return joint


def build_z1_urdf() -> ET.ElementTree:
    tree = ET.parse(URDF_DIR / "quadruped_z1.urdf")
    robot = tree.getroot()
    robot.set("name", "quadruped_z1")
    robot.insert(
        0,
        ET.Comment(
            " Go2 plus Unitree Z1. Joint origins, inertia, and limits are from "
            "unitreerobotics/unitree_ros robots/z1_description (BSD-3-Clause). "
            "Collision shapes are cylinders instead of the official meshes. "
            "Arm mass including the gripper is about 5.2 kg. "
        ),
    )
    for child in list(robot):
        name = child.get("name", "")
        if name.startswith("z1_") or name == "arm_mount_joint":
            robot.remove(child)
    arm = get_arm("z1")
    mount = ET.Element("joint", {"name": "arm_mount_joint", "type": "fixed"})
    ET.SubElement(
        mount,
        "origin",
        {"xyz": f"{MOUNT_XYZ[0]:.6f} {MOUNT_XYZ[1]:.6f} {MOUNT_XYZ[2]:.6f}", "rpy": "0 0 0"},
    )
    ET.SubElement(mount, "parent", {"link": "base"})
    ET.SubElement(mount, "child", {"link": "z1_link00"})
    robot.append(mount)

    shapes = {
        "z1_link00": (0.0325, 0.051, "0 0 0.0255", "0 0 0"),
        "z1_link01": (0.0325, 0.045, "0 0 0.02", "0 0 0"),
        "z1_link02": (0.022, 0.35, "-0.175 0 0", f"0 {math.pi / 2:.6f} 0"),
        "z1_link03": (0.020, 0.22, "0.11 0 0.03", f"0 {math.pi / 2:.6f} 0"),
        "z1_link04": (0.028, 0.07, "0.035 0 0", f"0 {math.pi / 2:.6f} 0"),
        "z1_link05": (0.025, 0.05, "0.025 0 0", f"0 {math.pi / 2:.6f} 0"),
        "z1_link06": (0.025, 0.04, "0.02 0 0", f"0 {math.pi / 2:.6f} 0"),
    }
    by_name = {item[0]: item for item in LINKS}
    for name, radius, length, xyz, rpy in (
        (key, *shapes[key]) for key in shapes
    ):
        _mass, _com, _inertia = by_name[name][1:]
        link = ET.Element("link", {"name": name})
        link.append(_inertial(by_name[name][1], by_name[name][2], by_name[name][3]))
        _cylinder(link, radius, length, xyz, rpy)
        robot.append(link)
    for name, parent, child, xyz, axis in JOINTS:
        lower, upper, effort = arm.limits[name]
        robot.append(_revolute(name, parent, child, xyz, axis, lower, upper, effort))

    stator = ET.Element("link", {"name": "z1_gripper_stator"})
    stator.append(_inertial(*by_name["z1_gripper_stator"][1:]))
    _box(stator, "0.06 0.04 0.03")
    robot.append(stator)
    fixed = ET.Element("joint", {"name": "z1_gripper_stator_joint", "type": "fixed"})
    ET.SubElement(fixed, "origin", {"xyz": "0.051 0 0", "rpy": "0 0 0"})
    ET.SubElement(fixed, "parent", {"link": "z1_link06"})
    ET.SubElement(fixed, "child", {"link": "z1_gripper_stator"})
    robot.append(fixed)

    mover = ET.Element("link", {"name": "z1_gripper_mover"})
    mover.append(_inertial(*by_name["z1_gripper_mover"][1:]))
    _box(mover, "0.04 0.02 0.02")
    robot.append(mover)
    lower, upper, effort = arm.limits["z1_jointGripper"]
    robot.append(_revolute("z1_jointGripper", "z1_gripper_stator", "z1_gripper_mover", "0.049 0 0", "0 1 0", lower, upper, effort))

    tcp = ET.Element("link", {"name": "z1_tcp"})
    tcp.append(_inertial(0.001, (0.0, 0.0, 0.0), (1e-6, 0.0, 0.0, 1e-6, 0.0, 1e-6)))
    robot.append(tcp)
    tcp_joint = ET.Element("joint", {"name": "z1_tcp_joint", "type": "fixed"})
    ET.SubElement(tcp_joint, "origin", {"xyz": "0.07 0 0", "rpy": "0 0 0"})
    ET.SubElement(tcp_joint, "parent", {"link": "z1_gripper_stator"})
    ET.SubElement(tcp_joint, "child", {"link": "z1_tcp"})
    robot.append(tcp_joint)
    return tree


def main() -> None:
    out = URDF_DIR / "quadruped_z1.urdf"
    tree = build_z1_urdf()
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
