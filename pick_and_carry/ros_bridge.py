"""Local socket between inference and a ROS 2 process.

This module does not import ROS. The simulator connects outward, sends one JSON
state line per step, and keeps the latest goal click. rviz2 listens inside WSL,
because the WSL firewall drops connections that start in WSL and arrive on
Windows. A missing or slow peer never blocks the physics step.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

from pick_and_carry.terrain import COURSE_X, COURSE_Y

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11411


def clamp_goal_xy(x: float, y: float) -> tuple[float, float]:
    """Keep a drop point inside the training course."""
    cx = min(float(COURSE_X[1]), max(float(COURSE_X[0]), float(x)))
    cy = min(float(COURSE_Y[1]), max(float(COURSE_Y[0]), float(y)))
    return cx, cy


def quat_wxyz_to_xyzw(quat: list[float]) -> list[float]:
    """Genesis stores w, x, y, z. ROS TF stores x, y, z, w."""
    w, x, y, z = (float(value) for value in quat)
    return [x, y, z, w]


def ros_usage_error(*, enabled: bool, play: bool, num_envs: int) -> str | None:
    """Why --ros cannot start, or None when the flags are valid."""
    if not enabled:
        return None
    if not play:
        return "--ros は推論（--play）のときだけ使えます。"
    if num_envs != 1:
        return "--ros は 1 体の推論だけです。--num-envs 1 を付けてください。"
    return None


def encode_message(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def split_messages(buffer: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """Split a TCP buffer into JSON objects. The remainder may be a partial line."""
    messages: list[dict[str, Any]] = []
    while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            messages.append(payload)
    return messages, buffer


def goal_from_message(payload: dict[str, Any]) -> tuple[float, float] | None:
    if payload.get("type") != "goal":
        return None
    try:
        return clamp_goal_xy(float(payload["x"]), float(payload["y"]))
    except (KeyError, TypeError, ValueError):
        return None


class InferenceBridge:
    """Connect to rviz2 and remember the newest goal. Call from the play loop."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._conn: socket.socket | None = None
        self._pending: bytes | None = None
        self._goal: tuple[float, float] | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="inference-ros", daemon=True)
        self._thread.start()

    def publish(self, state: dict[str, Any]) -> None:
        """Queue the latest state. Drops the previous line if it was not sent yet."""
        line = encode_message(state)
        with self._lock:
            self._pending = line

    def poll_goal(self) -> tuple[float, float] | None:
        """Return the newest goal once, then forget it."""
        with self._lock:
            goal = self._goal
            self._goal = None
        return goal

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        conn: socket.socket | None = None
        buf = b""
        while not self._stop.is_set():
            if conn is None:
                conn = self._connect()
                buf = b""
                if conn is None:
                    self._stop.wait(0.3)
                continue
            if not self._send_pending(conn):
                conn = self._drop(conn)
                continue
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                conn = self._drop(conn)
                continue
            if not chunk:
                conn = self._drop(conn)
                continue
            messages, buf = split_messages(buf + chunk)
            goal = self._latest_goal(messages)
            if goal is not None:
                with self._lock:
                    self._goal = goal

    def _connect(self) -> socket.socket | None:
        try:
            conn = socket.create_connection((self.host, self.port), timeout=0.5)
        except OSError:
            return None
        conn.settimeout(0.01)
        with self._lock:
            self._conn = conn
        return conn

    def _send_pending(self, conn: socket.socket) -> bool:
        with self._lock:
            outgoing = self._pending
            self._pending = None
        if outgoing is None:
            return True
        try:
            conn.sendall(outgoing)
        except OSError:
            return False
        return True

    def _drop(self, conn: socket.socket) -> None:
        with self._lock:
            if self._conn is conn:
                self._conn = None
        try:
            conn.close()
        except OSError:
            pass
        return None

    @staticmethod
    def _latest_goal(messages: list[dict[str, Any]]) -> tuple[float, float] | None:
        goal = None
        for payload in messages:
            parsed = goal_from_message(payload)
            if parsed is not None:
                goal = parsed
        return goal
