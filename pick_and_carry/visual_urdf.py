"""Swap primitive shapes for the official Go2 and Panda meshes.

Collision and inertia stay on the primitive URDF, so a trained policy still
sees the same body. Only the pictures change. Meshes ship with Genesis and are
not copied into this repo.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from importlib.util import find_spec
from pathlib import Path

from pick_and_carry.config import PRIMITIVE_URDF, URDF_DIR

VISUAL_URDF = URDF_DIR / ".visual" / "quadruped_arm.urdf"

# Link, mesh file relative to the Go2 dae folder or the Panda mesh root, visual rpy.
_GO2_VISUALS = {
    "base": ("base.dae", "0 0 0"),
    "FL_hip": ("hip.dae", "0 0 0"),
    "FR_hip": ("hip.dae", "3.1415 0 0"),
    "RL_hip": ("hip.dae", "0 3.1415 0"),
    "RR_hip": ("hip.dae", "3.1415 3.1415 0"),
    "FL_thigh": ("thigh.dae", "0 0 0"),
    "RL_thigh": ("thigh.dae", "0 0 0"),
    "FR_thigh": ("thigh_mirror.dae", "0 0 0"),
    "RR_thigh": ("thigh_mirror.dae", "0 0 0"),
    "FL_calf": ("calf.dae", "0 0 0"),
    "RL_calf": ("calf.dae", "0 0 0"),
    "FR_calf": ("calf_mirror.dae", "0 0 0"),
    "RR_calf": ("calf_mirror.dae", "0 0 0"),
    "FL_foot": ("foot.dae", "0 0 0"),
    "FR_foot": ("foot.dae", "0 0 0"),
    "RL_foot": ("foot.dae", "0 0 0"),
    "RR_foot": ("foot.dae", "0 0 0"),
}

_PANDA_VISUALS = {
    "panda_link0": ("collision/link0.obj", "0 0 0"),
    "panda_link1": ("visual/link1.obj", "0 0 0"),
    "panda_link2": ("visual/link2.obj", "0 0 0"),
    "panda_link3": ("visual/link3.obj", "0 0 0"),
    "panda_link4": ("visual/link4.obj", "0 0 0"),
    "panda_link5": ("visual/link5.obj", "0 0 0"),
    "panda_link6": ("visual/link6.obj", "0 0 0"),
    "panda_link7": ("collision/link7.obj", "0 0 0"),
    "panda_hand": ("visual/hand.obj", "0 0 0"),
    "panda_leftfinger": ("visual/finger.obj", "0 0 0"),
    "panda_rightfinger": ("visual/finger.obj", "0 0 3.14159265359"),
}


def _genesis_urdf_root() -> Path | None:
    spec = find_spec("genesis")
    if spec is None or not spec.origin:
        return None
    root = Path(spec.origin).parent / "assets" / "urdf"
    go2 = root / "go2" / "dae" / "base.dae"
    panda = root / "panda_bullet" / "meshes" / "visual" / "link1.obj"
    if not go2.is_file() or not panda.is_file():
        return None
    return root


def _mesh_visual(filename: Path, rpy: str) -> ET.Element:
    visual = ET.Element("visual")
    origin = ET.SubElement(visual, "origin")
    origin.set("xyz", "0 0 0")
    origin.set("rpy", rpy)
    mesh = ET.SubElement(ET.SubElement(visual, "geometry"), "mesh")
    mesh.set("filename", filename.resolve().as_posix())
    material = ET.SubElement(visual, "material")
    material.set("name", "")
    color = ET.SubElement(material, "color")
    color.set("rgba", "1 1 1 1")
    return visual


def styled_urdf_path() -> Path | None:
    """Write a visual URDF and return it, or None when the meshes are absent."""
    root = _genesis_urdf_root()
    if root is None or not PRIMITIVE_URDF.is_file():
        return None
    go2 = root / "go2" / "dae"
    panda = root / "panda_bullet" / "meshes"
    robot = ET.parse(PRIMITIVE_URDF).getroot()
    specs = {name: (go2 / filename, rpy) for name, (filename, rpy) in _GO2_VISUALS.items()}
    specs.update({name: (panda / filename, rpy) for name, (filename, rpy) in _PANDA_VISUALS.items()})
    for link in robot.findall("link"):
        spec = specs.get(link.get("name") or "")
        if spec is None:
            continue
        mesh_path, rpy = spec
        if not mesh_path.is_file():
            return None
        for visual in list(link.findall("visual")):
            link.remove(visual)
        link.append(_mesh_visual(mesh_path, rpy))
    VISUAL_URDF.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(VISUAL_URDF, encoding="utf-8", xml_declaration=True)
    return VISUAL_URDF
