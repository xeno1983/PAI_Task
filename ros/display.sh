#!/usr/bin/env bash
# Open rviz2 for inference. The Windows launcher starts this in WSL.
#   bash ros/display.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${DISPLAY:-}" && -S /mnt/wslg/.X11-unix/X0 ]]; then
  export DISPLAY=:0
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
  export PULSE_SERVER="${PULSE_SERVER:-/mnt/wslg/PulseServer}"
fi
ARM="${1:-z1}"
case "$ARM" in
  z1) URDF="$ROOT/assets/urdf/quadruped_z1.urdf" ;;
  *)
    echo "腕は z1 です: $ARM" >&2
    exit 1
    ;;
esac
if [[ ! -f "$URDF" ]]; then
  echo "URDF がありません: $URDF" >&2
  exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  for distro in jazzy humble iron; do
    if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "/opt/ros/${distro}/setup.bash"
      set -u
      break
    fi
  done
fi
if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "ROS 2 が見つかりません。desktop を入れてから source /opt/ros/<distro>/setup.bash してください。" >&2
  exit 1
fi

python3 - "$URDF" <<'PY'
import pathlib, re, sys
requested = pathlib.Path(sys.argv[1])
visual = requested.parent / ".visual" / requested.name
source = visual if visual.is_file() else requested
text = source.read_text(encoding="utf-8")
text = text.replace("\\", "/")
# rviz reads meshes with libcurl, so a plain /mnt/c path is rejected.
text = re.sub(
    r"([A-Za-z]):/",
    lambda match: "file:///mnt/" + match.group(1).lower() + "/",
    text,
)
body = ["  ros__parameters:", "    robot_description: |"]
for line in text.splitlines():
    body.append("      " + line)
block = "\n".join(body)
dumped = "/**:\n" + block + "\nrobot_state_publisher:\n" + block + "\n"
pathlib.Path("/tmp/pick_and_carry_rsp.yaml").write_text(dumped, encoding="utf-8")
print(source)
PY

LINK="$ROOT/logs/ros_ipc"
mkdir -p "$LINK"
rm -f "$LINK"/state.json "$LINK"/state-*.json "$LINK"/goal.json "$LINK"/goal-*.json "$LINK"/*.tmp
echo "腕: $ARM"
echo "URDF: $URDF"
echo "DISPLAY: ${DISPLAY:-未設定}"
echo "姿勢と運び先は logs/ros_ipc のファイルで受け渡します。Publish Point のクリックは黄色い球になり、推論中に緑の柱へ反映されます。"

ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/pick_and_carry_rsp.yaml &
RSP_PID=$!
python3 "$ROOT/ros/bridge_node.py" --link-dir "$LINK" &
BRIDGE_PID=$!

cleanup() {
  kill "$BRIDGE_PID" "$RSP_PID" 2>/dev/null || true
}
trap cleanup EXIT

rviz2 -d "$ROOT/ros/pick_and_carry.rviz"
