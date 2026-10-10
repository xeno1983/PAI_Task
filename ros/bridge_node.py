#!/usr/bin/env python3
"""ROS 2 side of inference. Run this with the sourced ROS Python, not the project venv.

Subscribes to /clicked_point and sends that XY as the drop point.
Publishes /joint_states, world -> base on /tf, and /markers.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

import tf2_geometry_msgs  # noqa: F401  registers PointStamped with the buffer


def _color(red: float, green: float, blue: float) -> ColorRGBA:
    return ColorRGBA(r=red, g=green, b=blue, a=0.9)


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
    def __init__(self, host: str, port: int) -> None:
        super().__init__("pick_and_carry_bridge")
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._server: socket.socket | None = None
        self._latest: dict | None = None
        self._announced = False
        self._joints = self.create_publisher(JointState, "/joint_states", 10)
        self._markers = self.create_publisher(MarkerArray, "/markers", 10)
        self._tf = TransformBroadcaster(self)
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)
        self.create_subscription(PointStamped, "/clicked_point", self._on_click, 10)
        self.create_timer(0.02, self._pump)
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

    def _pump(self) -> None:
        with self._lock:
            state = self._latest
        if not state:
            return
        if not self._announced:
            self._announced = True
            self.get_logger().info(
                f"腕 {state.get('arm')}  URDF {state.get('urdf')}  関節 {len(state.get('joint_names', []))}"
            )
        now = self.get_clock().now().to_msg()
        joints = JointState()
        joints.header.stamp = now
        joints.name = [str(name) for name in state["joint_names"]]
        joints.position = [float(value) for value in state["joint_pos"]]
        self._joints.publish(joints)

        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = "world"
        transform.child_frame_id = "base"
        base = state["base_pos"]
        quat = state["base_quat_wxyz"]
        transform.transform.translation.x = float(base[0])
        transform.transform.translation.y = float(base[1])
        transform.transform.translation.z = float(base[2])
        # Simulator quaternion is w, x, y, z. TF wants x, y, z, w.
        transform.transform.rotation.x = float(quat[1])
        transform.transform.rotation.y = float(quat[2])
        transform.transform.rotation.z = float(quat[3])
        transform.transform.rotation.w = float(quat[0])
        self._tf.sendTransform(transform)

        cube = float(state.get("cube_size", 0.04))
        array = MarkerArray()
        array.markers = [
            _marker(0, now, state["object_pos"], (cube, cube, cube), _color(0.88, 0.44, 0.19), Marker.CUBE),
            _marker(1, now, state["goal_pos"], (0.30, 0.30, 0.01), _color(0.18, 0.60, 0.33), Marker.CUBE),
            _marker(2, now, state["ee_pos"], (0.04, 0.04, 0.04), _color(0.20, 0.45, 0.85), Marker.SPHERE),
        ]
        self._markers.publish(array)

    def _on_click(self, point: PointStamped) -> None:
        frame = point.header.frame_id or "world"
        if frame not in ("world", "/world"):
            try:
                point = self._buffer.transform(point, "world", timeout=Duration(seconds=0.1))
            except Exception as exc:
                self.get_logger().warn(f"クリックを world に変換できません ({frame}): {exc}")
                return
        payload = json.dumps({"type": "goal", "x": float(point.point.x), "y": float(point.point.y)}) + "\n"
        with self._lock:
            sock = self._sock
        if sock is None:
            self.get_logger().warn("シミュレーション未接続のため、運び先を送れません。")
            return
        try:
            sock.sendall(payload.encode("utf-8"))
        except OSError as exc:
            self.get_logger().warn(f"運び先を送れません: {exc}")
        else:
            self.get_logger().info(f"運び先 ({point.point.x:.2f}, {point.point.y:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish inference state to rviz2 and forward goal clicks.")
    parser.add_argument("--sim-host", default="0.0.0.0", help="Address this node listens on.")
    parser.add_argument("--sim-port", type=int, default=11411)
    args = parser.parse_args()
    rclpy.init()
    node = PickAndCarryBridge(args.sim_host, args.sim_port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
