"""Weld a local go2.urdf and panda.urdf into quadruped_arm_merged.urdf."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pick_and_carry.config import MERGED_URDF, MOUNT_RPY, MOUNT_XYZ  # noqa: E402
from pick_and_carry.urdf_tools import UrdfError, merge_urdfs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix Franka onto the Go2 base and write one URDF.")
    parser.add_argument("--go2", type=Path, required=True, help="Path to go2.urdf")
    parser.add_argument("--panda", type=Path, required=True, help="Path to panda.urdf")
    parser.add_argument("--output", type=Path, default=MERGED_URDF)
    parser.add_argument("--parent-link", default=None, help="Go2 link that receives the arm. Default: base, else base_link, else trunk.")
    parser.add_argument("--arm-root", default=None, help="Arm link to weld. Default: the arm URDF root (usually panda_link0).")
    parser.add_argument("--mount-xyz", type=float, nargs=3, default=MOUNT_XYZ, metavar=("X", "Y", "Z"))
    parser.add_argument("--mount-rpy", type=float, nargs=3, default=MOUNT_RPY, metavar=("R", "P", "Y"))
    parser.add_argument(
        "--arm-mass-scale",
        type=float,
        default=1.0,
        help="Scale arm inertials. Try 0.25 if the 18 kg Franka pulls the 15 kg Go2 over.",
    )
    args = parser.parse_args()
    try:
        report = merge_urdfs(
            args.go2,
            args.panda,
            args.output,
            parent_link=args.parent_link,
            arm_root=args.arm_root,
            mount_xyz=tuple(args.mount_xyz),
            mount_rpy=tuple(args.mount_rpy),
            arm_mass_scale=args.arm_mass_scale,
        )
    except UrdfError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {report['output']}")
    print(f"mount {report['mount_parent']} -> {report['arm_root']} xyz={report['mount_xyz']} rpy={report['mount_rpy']}")
    print(f"links={report['n_links']} joints={report['n_joints']} root={report['root']}")
    missing = report["missing_meshes"]
    if missing:
        print(f"Unresolved meshes ({len(missing)}). First few:")
        for name in missing[:8]:
            print(f"  {name}")
    if report["missing_actuated_joints"] or report["missing_links"]:
        print("Actuated names not all present. The env looks up joints by name and will say which one is missing.")
        print(" missing joints:", report["missing_actuated_joints"])
        print(" missing links:", report["missing_links"])


if __name__ == "__main__":
    main()
