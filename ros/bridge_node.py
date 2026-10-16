#!/usr/bin/env python3
"""ROS 2 side of inference. Run this with the sourced ROS Python, not the project venv.

Subscribes to /clicked_point and sends that XY as the drop point.
Publishes /joint_states, world -> base on /tf, and /markers.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

import tf2_geometry_msgs  # noqa: F401  registers PointStamped with the buffer


# Standing pose so the model is not drawn with the calves outside their limits.
STANDING_JOINTS = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
    "z1_joint1": 0.0,
    "z1_joint2": 1.76,
    "z1_joint3": -1.0,
    "z1_joint4": 0.0,
    "z1_joint5": 0.0,
    "z1_joint6": 0.0,
    "z1_jointGripper": -0.4,
}


def _color(red: float, green: float, blue: float, alpha: float = 0.9) -> ColorRGBA:
    return ColorRGBA(r=red, g=green, b=blue, a=alpha)


def _marker(marker_id: int, stamp, position, scale, color, shape) -> Marker:
    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = "world"
    marker.ns = "pick_and_carry"
    marker.id = marker_id
    marker.type = shape
    marker.action = Marker.ADD
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    marker.pose.orientation.w = 1.0
    marker.scale.x = float(scale[0])
    marker.scale.y = float(scale[1])
    marker.scale.z = float(scale[2])
    marker.color = color
    return marker


class PickAndCarryBridge(Node):
    def __init__(self, host: str, port: int, link_dir: str = "") -> None:
        super().__init__("pick_and_carry_bridge")
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._server: socket.socket | None = None
        self._latest: dict | None = None
        self._announced = False
        self._pending_goal: bytes | None = None
        self._click_xy: tuple[float, float] | None = None
        self._last_goal_send = 0.0
        self._link_dir = Path(link_dir) if link_dir else None
        self._state_seq: object = None
        self._joints = self.create_publisher(JointState, "/joint_states", 10)
        self._image = self.create_publisher(Image, "/overview/image", 1)
        self._markers = self.create_publisher(MarkerArray, "/markers", 10)
        self._tf = TransformBroadcaster(self)
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)
        self.create_subscription(PointStamped, "/clicked_point", self._on_click, 10)
        self.create_timer(0.02, self._pump)
        if self._link_dir is not None:
            self.get_logger().info(f"姿勢ファイルを見ます {self._link_dir}")
            return
        self._listen(host, port)
        threading.Thread(target=self._accept_loop, name="sim-socket", daemon=True).start()

    def _listen(self, host: str, port: int) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        server.settimeout(0.5)
        self._server = server
        self.get_logger().info(f"推論からの接続を待っています {host}:{port}")

    def _accept_loop(self) -> None:
        while rclpy.ok():
            server = self._server
            if server is None:
                return
            try:
                sock, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            sock.settimeout(0.5)
            with self._lock:
                previous = self._sock
                self._sock = sock
            if previous is not None:
                try:
                    previous.close()
                except OSError:
                    pass
            self.get_logger().info("シミュレーションが接続しました")
            self._read(sock)
            with self._lock:
                if self._sock is sock:
                    self._sock = None
            self.get_logger().info("シミュレーションとの接続が切れました。再接続を待ちます。")

    def _read(self, sock: socket.socket) -> None:
        buf = b""
        try:
            while rclpy.ok():
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict) and payload.get("type") == "state":
                        with self._lock:
                            self._latest = payload
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _read_link(self) -> None:
        if self._link_dir is None:
            return
        newest = None
        newest_seq = None
        try:
            paths = list(self._link_dir.glob("state-*.json"))
        except OSError:
            return
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("type") != "state":
                continue
            try:
                seq = int(payload["seq"])
            except (KeyError, TypeError, ValueError):
                continue
            if newest_seq is None or seq > newest_seq:
                newest_seq = seq
                newest = payload
        if newest is None or newest_seq == self._state_seq:
            return
        self._state_seq = newest_seq
        with self._lock:
            self._latest = newest

    def _write_goal_file(self, x: float, y: float) -> None:
        assert self._link_dir is not None
        seq = 1
        for path in self._link_dir.glob("goal-*.json"):
            try:
                seq = max(seq, int(path.stem.split("-", 1)[1]) + 1)
            except (IndexError, ValueError):
                continue
        path = self._link_dir / f"goal-{seq}.json"
        body = json.dumps({"type": "goal", "x": x, "y": y, "seq": seq}).encode("utf-8")
        with path.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        stale = self._link_dir / f"goal-{seq - 5}.json"
        try:
            stale.unlink()
        except OSError:
            pass

    def _pump(self) -> None:
        self._read_link()
        with self._lock:
            state = self._latest
            click = self._click_xy
        now = self.get_clock().now().to_msg()
        if state is not None:
            self._acknowledge(state)
        self._flush_goal()
        if not state:
            # Keep world in the TF tree before the simulator connects, or rviz stays in error.
            self._send_base(now, (0.0, 0.0, 0.42), (1.0, 0.0, 0.0, 0.0))
            self._publish_standing(now)
            self._publish_click(now, click)
            return
        if not self._announced:
            self._announced = True
            self.get_logger().info(
                f"腕 {state.get('arm')}  URDF {state.get('urdf')}  関節 {len(state.get('joint_names', []))}"
            )
        joints = JointState()
        joints.header.stamp = now
        joints.name = [str(name) for name in state["joint_names"]]
        joints.position = [float(value) for value in state["joint_pos"]]
        self._joints.publish(joints)

        base = state["base_pos"]
        quat = state["base_quat_wxyz"]
        self._send_base(now, base, quat)

        cube = float(state.get("cube_size", 0.04))
        array = MarkerArray()
        array.markers = [
            self._course_marker(now),
            _marker(0, now, state["object_pos"], (cube, cube, cube), _color(0.88, 0.44, 0.19), Marker.CUBE),
            _marker(1, now, (state["goal_pos"][0], state["goal_pos"][1], float(state["goal_pos"][2]) + 0.20), (0.30, 0.30, 0.40), _color(0.15, 0.85, 0.30), Marker.CUBE),
            _marker(2, now, state["ee_pos"], (0.04, 0.04, 0.04), _color(0.20, 0.45, 0.85), Marker.SPHERE),
        ]
        if click is not None:
            array.markers.append(
                _marker(3, now, (click[0], click[1], 0.12), (0.18, 0.18, 0.18), _color(0.95, 0.85, 0.15), Marker.SPHERE)
            )
        self._markers.publish(array)
        try:
            self._publish_overview(state, now)
        except Exception as exc:
            self.get_logger().warn(f"俯瞰映像を出せません: {exc}")

    def _course_marker(self, stamp) -> Marker:
        # Covers the training course so Publish Point can hit it, not only the small grid.
        return _marker(4, stamp, (2.0, 0.0, 0.01), (5.6, 2.4, 0.02), _color(0.62, 0.55, 0.42, 0.45), Marker.CUBE)

    def _publish_standing(self, stamp) -> None:
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = list(STANDING_JOINTS)
        joints.position = [float(value) for value in STANDING_JOINTS.values()]
        self._joints.publish(joints)

    def _acknowledge(self, state: dict) -> None:
        click = self._click_xy
        goal = state.get("goal_pos")
        if click is None or not isinstance(goal, (list, tuple)) or len(goal) < 2:
            return
        if (float(goal[0]) - click[0]) ** 2 + (float(goal[1]) - click[1]) ** 2 <= 0.08 ** 2:
            with self._lock:
                self._pending_goal = None

    def _send_base(self, stamp, position, quat_wxyz) -> None:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "world"
        transform.child_frame_id = "base"
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])
        # Simulator quaternion is w, x, y, z. TF wants x, y, z, w.
        transform.transform.rotation.x = float(quat_wxyz[1])
        transform.transform.rotation.y = float(quat_wxyz[2])
        transform.transform.rotation.z = float(quat_wxyz[3])
        transform.transform.rotation.w = float(quat_wxyz[0])
        self._tf.sendTransform(transform)

    def _publish_click(self, stamp, click: tuple[float, float] | None) -> None:
        markers = [self._course_marker(stamp)]
        if click is not None:
            markers.append(
                _marker(3, stamp, (click[0], click[1], 0.12), (0.18, 0.18, 0.18), _color(0.95, 0.85, 0.15), Marker.SPHERE)
            )
        array = MarkerArray()
        array.markers = markers
        self._markers.publish(array)

    def _flush_goal(self) -> None:
        now = time.monotonic()
        with self._lock:
            payload = self._pending_goal
            sock = self._sock
            if payload is None or sock is None or now - self._last_goal_send < 0.2:
                return
        try:
            sock.sendall(payload)
        except OSError as exc:
            self.get_logger().warn(f"運び先を送れません: {exc}")
            return
        self._last_goal_send = now

    def _publish_overview(self, state: dict, stamp) -> None:
        packed = state.get("image")
        if not isinstance(packed, dict) or "data" not in packed:
            return
        width = int(packed["w"])
        height = int(packed["h"])
        raw = base64.b64decode(packed["data"])
        if len(raw) != width * height * 3:
            return
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = "world"
        image.height = height
        image.width = width
        image.encoding = "rgb8"
        image.is_bigendian = 0
        image.step = width * 3
        image.data = raw
        self._image.publish(image)

    def _on_click(self, point: PointStamped) -> None:
        frame = point.header.frame_id or "world"
        if frame not in ("world", "/world"):
            try:
                point = self._buffer.transform(point, "world", timeout=Duration(seconds=0.1))
            except Exception as exc:
                self.get_logger().warn(f"クリックを world に変換できません ({frame}): {exc}")
                return
        x = float(point.point.x)
        y = float(point.point.y)
        with self._lock:
            self._click_xy = (x, y)
        self.get_logger().info(f"運び先 ({x:.2f}, {y:.2f})")
        if self._link_dir is not None:
            try:
                self._write_goal_file(x, y)
            except OSError as exc:
                self.get_logger().warn(f"運び先を書けません: {exc}")
            return
        payload = (json.dumps({"type": "goal", "x": x, "y": y, "seq": time.time_ns()}) + "\n").encode("utf-8")
        with self._lock:
            self._pending_goal = payload
        self._flush_goal()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish inference state to rviz2 and forward goal clicks.")
    parser.add_argument("--sim-host", default="0.0.0.0", help="Address this node listens on.")
    parser.add_argument("--sim-port", type=int, default=11411)
    parser.add_argument("--link-dir", default="", help="Shared folder for state.json and goal.json.")
    args = parser.parse_args()
    rclpy.init()
    node = PickAndCarryBridge(args.sim_host, args.sim_port, args.link_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
