"""URDF merge and the committed primitive model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pick_and_carry.config import MOUNT_RPY, MOUNT_XYZ, PRIMITIVE_URDF
from pick_and_carry.urdf_tools import UrdfError, merge_urdfs, validate_urdf

GO2 = """<?xml version="1.0"?>
<robot name="go2">
  <link name="base">
    <inertial>
      <mass value="6"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual><geometry><mesh filename="package://go2_description/dae/base.dae"/></geometry></visual>
  </link>
  <link name="FL_foot"/>
  <joint name="FL_hip_joint" type="revolute">
    <parent link="base"/>
    <child link="FL_foot"/>
    <axis xyz="1 0 0"/>
  </joint>
</robot>
"""

PANDA = """<?xml version="1.0"?>
<robot name="panda" xmlns="http://www.ros.org">
  <link name="world"/>
  <link name="panda_link0">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>
  <joint name="world_joint" type="fixed">
    <parent link="world"/>
    <child link="panda_link0"/>
  </joint>
  <link name="panda_link1"/>
  <joint name="panda_joint1" type="revolute">
    <parent link="panda_link0"/>
    <child link="panda_link1"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""


class PrimitiveUrdfTests(unittest.TestCase):
    def test_committed_model(self):
        report = validate_urdf(PRIMITIVE_URDF, require_actuated_names=True, require_tcp=True)
        self.assertEqual(report["root"], "base")
        self.assertEqual(report["mount_parent"], "base")
        self.assertEqual(report["mount_child"], "panda_link0")
        for got, expected in zip(report["mount_xyz"], MOUNT_XYZ):
            self.assertAlmostEqual(got, expected, places=5)
        for got, expected in zip(report["mount_rpy"], MOUNT_RPY):
            self.assertAlmostEqual(got, expected, places=5)


class MergeTests(unittest.TestCase):
    def test_fixed_mount_and_mesh_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dae").mkdir()
            mesh = root / "dae" / "base.dae"
            mesh.write_text("solid", encoding="utf-8")
            go2 = root / "go2.urdf"
            panda = root / "panda.urdf"
            go2.write_text(GO2, encoding="utf-8")
            panda.write_text(PANDA, encoding="utf-8")
            output = root / "quadruped_arm_merged.urdf"
            report = merge_urdfs(go2, panda, output, arm_mass_scale=0.5)
            self.assertEqual(report["root"], "base")
            self.assertEqual(report["mount_parent"], "base")
            self.assertEqual(report["mount_child"], "panda_link0")
            text = output.read_text(encoding="utf-8")
            self.assertIn(mesh.resolve().as_posix(), text)
            self.assertNotIn("world_joint", text)
            self.assertIn('value="1"', text)

    def test_name_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            go2 = root / "go2.urdf"
            panda = root / "panda.urdf"
            go2.write_text(GO2, encoding="utf-8")
            panda.write_text(GO2.replace('name="go2"', 'name="panda"'), encoding="utf-8")
            with self.assertRaises(UrdfError):
                merge_urdfs(go2, panda, root / "out.urdf")


if __name__ == "__main__":
    unittest.main()
