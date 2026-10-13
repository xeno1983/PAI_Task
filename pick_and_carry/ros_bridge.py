"""Local socket between inference and a ROS 2 process.

This module does not import ROS. The simulator connects outward, sends one JSON
state line per step, and keeps the latest goal click. rviz2 listens inside WSL,
because the WSL firewall drops connections that start in WSL and arrive on
Windows. A missing or slow peer never blocks the physics step.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

import numpy as np

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


def downsample_rgb(frame, stride: int = 2) -> np.ndarray:
    """RGB uint8 image, taking every ``stride`` pixel."""
    image = np.asarray(frame)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] > 3:
        image = image[..., :3]
    image = np.ascontiguousarray(image[::stride, ::stride])
    if image.dtype != np.uint8:
        peak = float(image.max()) if image.size else 1.0
        if peak <= 1.0:
            image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
    return image


def pack_rgb_image(frame, stride: int = 2) -> dict[str, object]:
    """Downsample an RGB frame for the live rviz view. ``stride`` keeps the socket small."""
    image = downsample_rgb(frame, stride)
    height, width = int(image.shape[0]), int(image.shape[1])
    return {
        "w": width,
        "h": height,
        "encoding": "rgb8",
        "data": base64.b64encode(image.tobytes()).decode("ascii"),
    }


def goal_from_message(payload: dict[str, Any]) -> tuple[float, float] | None:
    if payload.get("type") != "goal":
        return None
    try:
        return clamp_goal_xy(float(payload["x"]), float(payload["y"]))
    except (KeyError, TypeError, ValueError):
        return None


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a new file. Replacing an open file is denied while rviz2 is reading it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class FileBridge:
    """Pass poses and goal clicks through a folder both Windows and WSL can see.

    TCP between the two is dropped by the Hyper-V firewall in both directions.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._state_seq = 0
        self._goal_seq: int | None = None
        self._warned = False

    def start(self) -> None:
        return None

    def publish(self, state: dict[str, Any]) -> None:
        body = {key: value for key, value in state.items() if key != "image"}
        self._state_seq += 1
        body["seq"] = self._state_seq
        path = self.directory / f"state-{self._state_seq}.json"
        try:
            write_new_json(path, body)
        except OSError as exc:
            self._state_seq -= 1
            if not self._warned:
                print(f"姿勢を rviz2 に渡せません。{exc}", flush=True)
                self._warned = True
            return
        self._drop_old("state", self._state_seq)

    def poll_goal(self) -> tuple[float, float] | None:
        """Return a goal when a newer goal file appears."""
        newest: dict[str, Any] | None = None
        newest_seq: int | None = None
        for path in self.directory.glob("goal-*.json"):
            payload = read_json(path)
            if payload is None:
                continue
            try:
                seq = int(payload["seq"])
            except (KeyError, TypeError, ValueError):
                continue
            if newest_seq is None or seq > newest_seq:
                newest_seq = seq
                newest = payload
        if newest is None or newest_seq is None or newest_seq == self._goal_seq:
            return None
        goal = goal_from_message(newest)
        if goal is None:
            return None
        self._goal_seq = newest_seq
        return goal

    def _drop_old(self, prefix: str, seq: int) -> None:
        stale = self.directory / f"{prefix}-{seq - 8}.json"
        try:
            stale.unlink()
        except OSError:
            pass
        if seq % 50 != 0:
            return
        for path in self.directory.glob(f"{prefix}-*.json"):
            try:
                older = int(path.stem.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if older < seq - 8:
                try:
                    path.unlink()
                except OSError:
                    pass

    def close(self) -> None:
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
        """Write on this thread and read on another, so a large frame cannot block a goal click."""
        while not self._stop.is_set():
            conn = self._connect()
            if conn is None:
                self._stop.wait(0.3)
                continue
            reader = threading.Thread(target=self._read_loop, args=(conn,), name="inference-ros-read", daemon=True)
            reader.start()
            while not self._stop.is_set() and self._same_connection(conn):
                if not self._send_pending(conn):
                    self._drop(conn)
                    break
                self._stop.wait(0.01)
            reader.join(timeout=0.5)

    def _same_connection(self, conn: socket.socket) -> bool:
        with self._lock:
            return self._conn is conn

    def _read_loop(self, conn: socket.socket) -> None:
        buf = b""
        while not self._stop.is_set() and self._same_connection(conn):
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            messages, buf = split_messages(buf + chunk)
            goal = self._latest_goal(messages)
            if goal is not None:
                with self._lock:
                    self._goal = goal
        self._drop(conn)

    def _connect(self) -> socket.socket | None:
        try:
            conn = socket.create_connection((self.host, self.port), timeout=0.5)
        except OSError:
            return None
        conn.settimeout(0.05)
        with self._lock:
            self._conn = conn
        return conn

    def _send_pending(self, conn: socket.socket) -> bool:
        with self._lock:
            outgoing = self._pending
        if outgoing is None:
            return True
        try:
            # The reader thread keeps taking goals while this send is in progress.
            # A timeout means the line was cut off, so the caller drops the socket.
            conn.settimeout(2.0)
            conn.sendall(outgoing)
            conn.settimeout(0.05)
        except OSError:
            return False
        with self._lock:
            if self._pending is outgoing:
                self._pending = None
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
