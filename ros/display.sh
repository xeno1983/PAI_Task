#!/usr/bin/env bash
# Open rviz2 for inference. The Windows launcher starts this in WSL.
#   bash ros/display.sh panda
#   bash ros/display.sh z1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${DISPLAY:-}" && -S /mnt/wslg/.X11-unix/X0 ]]; then
  export DISPLAY=:0
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
  export PULSE_SERVER="${PULSE_SERVER:-/mnt/wslg/PulseServer}"
fi
ARM="${1:-panda}"
case "$ARM" in
  z1) URDF="$ROOT/assets/urdf/quadruped_z1.urdf" ;;
  panda) URDF="$ROOT/assets/urdf/quadruped_arm.urdf" ;;
  *)
    echo "腕は panda または z1 です: $ARM" >&2
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
import pathlib, sys
urdf = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
body = ["  ros__parameters:", "    robot_description: |"]
for line in urdf.splitlines():
    body.append("      " + line)
block = "\n".join(body)
text = "/**:\n" + block + "\nrobot_state_publisher:\n" + block + "\n"
pathlib.Path("/tmp/pick_and_carry_rsp.yaml").write_text(text, encoding="utf-8")
PY

SIM_PORT="${SIM_PORT:-11411}"
echo "腕: $ARM"
echo "URDF: $URDF"
echo "DISPLAY: ${DISPLAY:-未設定}"
echo "推論からの接続を ${SIM_PORT} 番で待ちます。rviz2 の Publish Point で地面をクリックすると運び先が動きます。"

ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/pick_and_carry_rsp.yaml &
RSP_PID=$!
python3 "$ROOT/ros/bridge_node.py" --sim-host 0.0.0.0 --sim-port "$SIM_PORT" &
BRIDGE_PID=$!

cleanup() {
  kill "$BRIDGE_PID" "$RSP_PID" 2>/dev/null || true
}
trap cleanup EXIT

rviz2 -d "$ROOT/ros/pick_and_carry.rviz"
