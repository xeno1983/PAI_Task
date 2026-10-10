"""Check link and joint consistency of a quadruped-arm URDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pick_and_carry.config import PRIMITIVE_URDF  # noqa: E402
from pick_and_carry.urdf_tools import UrdfError, validate_urdf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate quadruped_arm URDF structure.")
    parser.add_argument("urdf", type=Path, nargs="?", default=PRIMITIVE_URDF)
    parser.add_argument("--no-actuated-check", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_urdf(args.urdf, require_actuated_names=not args.no_actuated_check, require_tcp=False)
    except UrdfError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"{report['path']}: root={report['root']} links={report['n_links']} joints={report['n_joints']} "
        f"mount={report['mount_parent']}->{report['mount_child']} xyz={report['mount_xyz']}"
    )


if __name__ == "__main__":
    main()
