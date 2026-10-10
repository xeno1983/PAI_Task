"""Merge a Go2 URDF and a Franka URDF, and check the combined tree."""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from pick_and_carry.config import (
    ARM_JOINTS,
    ARM_ROOT_LINK,
    BASE_LINK_CANDIDATES,
    FINGER_JOINTS,
    FOOT_LINKS,
    LEG_JOINTS,
    MOUNT_JOINT,
    MOUNT_RPY,
    MOUNT_XYZ,
)

WORLD_LINKS = {"world", "world_link"}
REQUIRED_JOINTS = (MOUNT_JOINT, *LEG_JOINTS, *ARM_JOINTS, *FINGER_JOINTS)
REQUIRED_LINKS = ("base", ARM_ROOT_LINK, *FOOT_LINKS)


class UrdfError(ValueError):
    pass


def parse_urdf(path: str | Path) -> ET.Element:
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r'\sxmlns(?::\w+)?="[^"]*"', "", text)
    root = ET.fromstring(text)
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]
    return root


def _direct(parent: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in list(parent) if child.tag == tag]


def _joint_ends(joint: ET.Element) -> tuple[str, str]:
    parent = joint.find("parent")
    child = joint.find("child")
    if parent is None or child is None or not parent.get("link") or not child.get("link"):
        raise UrdfError(f"Joint '{joint.get('name')}' is missing parent or child.")
    return parent.get("link"), child.get("link")


def link_names(robot: ET.Element) -> list[str]:
    return [link.get("name") for link in _direct(robot, "link")]


def joint_names(robot: ET.Element) -> list[str]:
    return [joint.get("name") for joint in _direct(robot, "joint")]


def root_links(robot: ET.Element) -> list[str]:
    children = { _joint_ends(joint)[1] for joint in _direct(robot, "joint") }
    return [name for name in link_names(robot) if name not in children]


def _detach_world(robot: ET.Element) -> None:
    """Drop a dummy world link so the arm root can be welded onto Go2."""
    drop_links = {name for name in root_links(robot) if name.lower() in WORLD_LINKS}
    if not drop_links:
        return
    for joint in list(_direct(robot, "joint")):
        parent, _child = _joint_ends(joint)
        if parent in drop_links:
            robot.remove(joint)
    for link in list(_direct(robot, "link")):
        if link.get("name") in drop_links:
            robot.remove(link)


def _resolve_mesh(filename: str, source_dir: Path) -> Path | None:
    raw = Path(filename)
    if raw.is_file():
        return raw.resolve()
    relative = filename
    if filename.startswith("package://"):
        rest = filename[len("package://") :]
        parts = rest.split("/", 1)
        relative = parts[1] if len(parts) == 2 else parts[0]
    search_roots = [source_dir, *list(source_dir.parents)[:6]]
    for root in search_roots:
        candidate = (root / relative).resolve()
        if candidate.is_file():
            return candidate
    plain = (source_dir / filename).resolve()
    if plain.is_file():
        return plain
    return None


def rewrite_mesh_paths(robot: ET.Element, source_dir: Path) -> list[str]:
    missing: list[str] = []
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        resolved = _resolve_mesh(filename, source_dir)
        if resolved is None:
            missing.append(filename)
            continue
        mesh.set("filename", resolved.as_posix())
    return missing


def _name_unnamed_materials(robot: ET.Element) -> None:
    """Genesis urdfpy rejects a <material> that has no name."""
    index = 0
    for material in robot.iter("material"):
        if material.get("name"):
            continue
        material.set("name", f"unnamed_material_{index}")
        index += 1


def _scale_inertials(robot: ET.Element, scale: float) -> None:
    if scale == 1.0:
        return
    for inertial in robot.iter("inertial"):
        for child in list(inertial):
            if child.tag == "mass" and child.get("value") is not None:
                child.set("value", f"{float(child.get('value')) * scale:.8g}")
            if child.tag == "inertia":
                for attr in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                    if child.get(attr) is not None:
                        child.set(attr, f"{float(child.get(attr)) * scale:.8g}")


def _rename_colliding_materials(arm: ET.Element, used: set[str]) -> None:
    collisions = {element.get("name") for element in arm.iter("material") if element.get("name") in used}
    if not collisions:
        return
    for material in arm.iter("material"):
        name = material.get("name")
        if name in collisions:
            material.set("name", f"franka_mat_{name}")


def _mount_joint(parent_link: str, child_link: str, xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> ET.Element:
    joint = ET.Element("joint", {"name": MOUNT_JOINT, "type": "fixed"})
    ET.SubElement(
        joint,
        "origin",
        {"xyz": f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}", "rpy": f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"},
    )
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})
    return joint


def _choose_parent(go2: ET.Element, parent_link: str | None) -> str:
    names = set(link_names(go2))
    if parent_link is not None:
        if parent_link not in names:
            raise UrdfError(f"Go2 URDF has no link '{parent_link}'. Links start with: {sorted(names)[:12]}")
        return parent_link
    for candidate in BASE_LINK_CANDIDATES:
        if candidate in names:
            return candidate
    raise UrdfError(
        "Could not find a Go2 base link. Pass --parent-link. "
        f"Looked for {BASE_LINK_CANDIDATES}. First links: {link_names(go2)[:12]}"
    )


def _choose_arm_root(arm: ET.Element, arm_root: str | None) -> str:
    roots = root_links(arm)
    if arm_root is not None:
        if arm_root not in link_names(arm):
            raise UrdfError(f"Arm URDF has no link '{arm_root}'.")
        if arm_root not in roots:
            raise UrdfError(f"'{arm_root}' is not a root of the arm URDF. Roots: {roots}")
        return arm_root
    if len(roots) != 1:
        raise UrdfError(f"Arm URDF must have one root after removing world links. Found {roots}.")
    return roots[0]


def merge_urdfs(
    go2_path: str | Path,
    panda_path: str | Path,
    output_path: str | Path,
    *,
    parent_link: str | None = None,
    arm_root: str | None = None,
    mount_xyz: tuple[float, float, float] = MOUNT_XYZ,
    mount_rpy: tuple[float, float, float] = MOUNT_RPY,
    arm_mass_scale: float = 1.0,
) -> dict[str, object]:
    """Weld the Franka root to the Go2 base with a fixed joint.

    Mesh filenames are rewritten to absolute paths so Colab can load the file
    without a ROS package layout. ``arm_mass_scale`` multiplies every inertial
    of the arm. A full Franka is about 18 kg; Go2 is about 15 kg, so early
    training often wants a scale around 0.25.
    """
    go2_path = Path(go2_path)
    panda_path = Path(panda_path)
    output_path = Path(output_path)

    go2 = parse_urdf(go2_path)
    arm = parse_urdf(panda_path)
    _detach_world(go2)
    _detach_world(arm)

    go2_links = set(link_names(go2))
    arm_links = set(link_names(arm))
    go2_joints = set(joint_names(go2))
    arm_joints = set(joint_names(arm))
    link_clash = sorted(go2_links & arm_links)
    joint_clash = sorted((go2_joints & arm_joints) | ({MOUNT_JOINT} & (go2_joints | arm_joints)))
    if link_clash or joint_clash:
        raise UrdfError(f"Name collision. links={link_clash} joints={joint_clash}")

    parent = _choose_parent(go2, parent_link)
    child = _choose_arm_root(arm, arm_root)
    if parent == child:
        raise UrdfError("Mount parent and arm root are the same link.")

    _name_unnamed_materials(go2)
    _name_unnamed_materials(arm)
    _rename_colliding_materials(arm, {element.get("name") for element in go2.iter("material") if element.get("name")})
    missing_go2 = rewrite_mesh_paths(go2, go2_path.parent)
    missing_arm = rewrite_mesh_paths(arm, panda_path.parent)
    _scale_inertials(arm, arm_mass_scale)

    merged = ET.Element("robot", {"name": "quadruped_arm"})
    merged.append(
        ET.Comment(
            " Go2 base to Franka panda base, fixed. "
            f"xyz={mount_xyz} rpy={mount_rpy}. "
            "+X is forward on Go2, +Z is up, arm stands upright. "
        )
    )
    for element in list(go2):
        merged.append(copy.deepcopy(element))
    merged.append(_mount_joint(parent, child, mount_xyz, mount_rpy))
    for element in list(arm):
        merged.append(copy.deepcopy(element))

    report = validate_robot(merged, require_actuated_names=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(merged)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    report.update(
        {
            "output": str(output_path),
            "parent_link": parent,
            "arm_root": child,
            "mount_xyz": mount_xyz,
            "mount_rpy": mount_rpy,
            "arm_mass_scale": arm_mass_scale,
            "missing_meshes": missing_go2 + missing_arm,
        }
    )
    return report


def validate_robot(robot: ET.Element, *, require_actuated_names: bool = True, require_tcp: bool = False) -> dict[str, object]:
    links = link_names(robot)
    joints = joint_names(robot)
    if any(name is None for name in links) or len(links) != len(set(links)):
        raise UrdfError("Duplicate or unnamed links.")
    if any(name is None for name in joints) or len(joints) != len(set(joints)):
        raise UrdfError("Duplicate or unnamed joints.")

    link_set = set(links)
    mount = None
    for joint in _direct(robot, "joint"):
        parent, child = _joint_ends(joint)
        if parent not in link_set or child not in link_set:
            raise UrdfError(f"Joint '{joint.get('name')}' references a missing link ({parent} -> {child}).")
        if joint.get("name") == MOUNT_JOINT:
            mount = joint
    roots = root_links(robot)
    if len(roots) != 1:
        raise UrdfError(f"Combined URDF must have exactly one root link. Found {roots}.")
    if mount is None:
        raise UrdfError(f"Missing fixed joint '{MOUNT_JOINT}'.")
    if mount.get("type") != "fixed":
        raise UrdfError(f"{MOUNT_JOINT} must be fixed, got '{mount.get('type')}'.")
    parent, child = _joint_ends(mount)

    missing_joints = [name for name in REQUIRED_JOINTS if name not in joints] if require_actuated_names else []
    missing_links = [name for name in REQUIRED_LINKS if name not in link_set]
    if require_tcp and "panda_hand_tcp" not in link_set:
        missing_links.append("panda_hand_tcp")
    if require_actuated_names and (missing_joints or missing_links):
        raise UrdfError(f"Missing joints {missing_joints} links {missing_links}.")

    origin = mount.find("origin")
    xyz = tuple(float(part) for part in origin.get("xyz", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
    rpy = tuple(float(part) for part in origin.get("rpy", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
    return {
        "root": roots[0],
        "n_links": len(links),
        "n_joints": len(joints),
        "mount_parent": parent,
        "mount_child": child,
        "mount_xyz": xyz,
        "mount_rpy": rpy,
        "missing_actuated_joints": missing_joints,
        "missing_links": missing_links,
    }


def validate_urdf(path: str | Path, **kwargs) -> dict[str, object]:
    report = validate_robot(parse_urdf(path), **kwargs)
    report["path"] = str(path)
    return report
