"""Height field for the rough-terrain pick-and-place course.

The robot starts on a flat pad, crosses a bumpy strip, picks the cube up on a
second flat pad, crosses another bumpy strip, and releases on a final flat pad.
Heights are in meters. Genesis multiplies the grid by ``vertical_scale``, so
the environment passes ``vertical_scale=1``.
"""

from __future__ import annotations

import numpy as np

# World-frame intervals where the mask is fully rough or fully flat.
ROUGH_1 = (0.55, 0.85)
PICKUP_FLAT = (1.05, 2.15)
ROUGH_2 = (2.40, 3.05)
GOAL_FLAT = (3.30, 4.55)
COURSE_X = (-0.8, 4.8)
COURSE_Y = (-1.2, 1.2)
# Spawn boxes sit inside the flat pads, far enough that pickup and release
# begin only after the body has left the bumps.
OBJECT_X_RANGE = (1.60, 1.95)
GOAL_X_RANGE = (3.75, 4.25)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    span = max(edge1 - edge0, 1e-6)
    t = np.clip((x - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rough_mask(x: np.ndarray) -> np.ndarray:
    """1 on the two travel strips, 0 on the start, pickup, and goal pads."""
    first = _smoothstep(0.35, ROUGH_1[0], x) * (1.0 - _smoothstep(ROUGH_1[1], PICKUP_FLAT[0], x))
    second = _smoothstep(PICKUP_FLAT[1], ROUGH_2[0], x) * (1.0 - _smoothstep(ROUGH_2[1], GOAL_FLAT[0], x))
    return np.clip(first + second, 0.0, 1.0)


# Flat gutter between side-by-side courses so one env cannot walk onto another's bumps.
TILE_GAP = 2.0


def _plateau(x: np.ndarray, start: float, end: float, ramp: float = 0.05) -> np.ndarray:
    """Flat tread with short ramps. Zero outside ``[start, end]``."""
    up = _smoothstep(start, start + ramp, x)
    down = 1.0 - _smoothstep(end - ramp, end, x)
    return np.clip(up * down, 0.0, 1.0)


def make_course(
    amplitude: float = 0.035,
    horizontal_scale: float = 0.05,
    rng: np.random.Generator | None = None,
    steps: bool = False,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Return ``(height[x, y] in meters, origin xy, horizontal scale)``.

    Without ``rng`` the bumps are the original fixed waves. With ``rng`` each call
    draws a new wavelength, phase, and height, while the pads stay flat.
    ``steps`` adds flat treads on the two travel strips: one before the box, two after it.
    """
    if amplitude < 0.0:
        raise ValueError("terrain amplitude must be non-negative")
    if horizontal_scale <= 0.0:
        raise ValueError("horizontal scale must be positive")
    x0, x1 = COURSE_X
    y0, y1 = COURSE_Y
    rows = int(round((x1 - x0) / horizontal_scale)) + 1
    cols = int(round((y1 - y0) / horizontal_scale)) + 1
    xs = x0 + np.arange(rows) * horizontal_scale
    ys = y0 + np.arange(cols) * horizontal_scale
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    if rng is None:
        wave = np.sin(2.0 * np.pi * grid_x / 0.50) * np.cos(2.0 * np.pi * grid_y / 0.70)
        wave += 0.45 * np.sin(2.0 * np.pi * grid_x / 0.28 + 0.8)
        height_scale = amplitude
    else:
        length_x = float(rng.uniform(0.36, 0.70))
        length_y = float(rng.uniform(0.45, 0.95))
        length_x2 = float(rng.uniform(0.22, 0.40))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        mix = float(rng.uniform(0.25, 0.65))
        wave = np.sin(2.0 * np.pi * grid_x / length_x + phase[0]) * np.cos(2.0 * np.pi * grid_y / length_y + phase[1])
        wave += mix * np.sin(2.0 * np.pi * grid_x / length_x2 + phase[2])
        height_scale = amplitude * float(rng.uniform(0.55, 1.0))
    if steps:
        shift = np.zeros(3) if rng is None else rng.uniform(-0.02, 0.02, size=3)
        tread = _plateau(grid_x, 0.57 + shift[0], 0.83 + shift[0])
        tread = np.maximum(tread, _plateau(grid_x, 2.45 + shift[1], 2.68 + shift[1]))
        tread = np.maximum(tread, _plateau(grid_x, 2.80 + shift[2], 3.02 + shift[2]))
        unit = np.clip(0.20 + 0.80 * tread + 0.12 * wave, 0.0, 1.0)
    else:
        unit = np.clip(0.55 + 0.35 * wave, 0.0, 1.0)
    height = (height_scale * rough_mask(grid_x) * unit).astype(np.float32)
    return height, (float(x0), float(y0)), float(horizontal_scale)


def make_terrain_bank(
    n_tiles: int,
    amplitude: float = 0.035,
    horizontal_scale: float = 0.05,
    rng: np.random.Generator | None = None,
    gap: float = TILE_GAP,
    steps: bool = False,
) -> tuple[np.ndarray, tuple[float, float], float, np.ndarray, int, int]:
    """Lay ``n_tiles`` courses along Y. Each reset can replace one tile in place.

    Returns height, origin xy, scale, shifts (n, 2) added to local course xy,
    columns per tile, and the column stride from one tile to the next.
    """
    if n_tiles < 1:
        raise ValueError("need at least one terrain tile")
    tile, origin, scale = make_course(amplitude, horizontal_scale, rng, steps=steps)
    rows, cols = tile.shape
    gap_cols = int(round(gap / scale))
    stride = cols + gap_cols
    total_cols = n_tiles * stride - gap_cols
    height = np.zeros((rows, total_cols), dtype=np.float32)
    shifts = np.zeros((n_tiles, 2), dtype=np.float32)
    for index in range(n_tiles):
        if index == 0:
            course = tile
        else:
            course, _, _ = make_course(amplitude, horizontal_scale, rng, steps=steps)
        column = index * stride
        height[:, column : column + cols] = course
        shifts[index, 1] = index * stride * scale
    return height, origin, scale, shifts, cols, stride


YARD_PITCH = 4.6


def yard_waypoints(n_stations: int) -> tuple[np.ndarray, np.ndarray]:
    """Pickup and drop points that snake left and right, so the body must turn.

    Each leg is a few meters. Consecutive headings differ by roughly a right-angle
    turn, while the path as a whole still runs toward +X.
    """
    if n_stations < 1:
        raise ValueError("need at least one supply station")
    pickups = np.zeros((n_stations, 2), dtype=np.float32)
    goals = np.zeros((n_stations, 2), dtype=np.float32)
    x = 2.3
    y = 1.2
    for index in range(n_stations):
        side = 1.0 if index % 2 == 0 else -1.0
        pickups[index, 0] = x
        pickups[index, 1] = y
        goals[index, 0] = x + 1.7
        goals[index, 1] = y + 0.7 * side
        x = x + 3.4
        y = -side * (1.2 + 0.35 * (index % 2))
    return pickups, goals


def make_supply_yard(
    n_stations: int,
    amplitude: float = 0.05,
    horizontal_scale: float = 0.05,
    pitch: float = YARD_PITCH,
    y_half: float = 3.0,
) -> tuple[np.ndarray, tuple[float, float], float, np.ndarray, np.ndarray]:
    """Wide lumpy yard. Flat disks sit under each box and each drop point.

    Returns height, origin xy, scale, pickup xy (n, 2), and goal xy (n, 2).
    ``pitch`` is unused; the stations follow :func:`yard_waypoints`.
    """
    del pitch
    pickups, goals = yard_waypoints(n_stations)
    span = np.concatenate((pickups, goals), axis=0)
    y_half = max(y_half, float(np.max(np.abs(span[:, 1]))) + 1.5)
    x0, x1 = -0.8, float(span[:, 0].max()) + 1.6
    y0, y1 = -y_half, y_half
    rows = int(round((x1 - x0) / horizontal_scale)) + 1
    cols = int(round((y1 - y0) / horizontal_scale)) + 1
    xs = x0 + np.arange(rows) * horizontal_scale
    ys = y0 + np.arange(cols) * horizontal_scale
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    rough = np.ones(grid_x.shape, dtype=np.float32)
    pads = [(0.2, 0.0, 0.95), *((float(point[0]), float(point[1]), 1.05) for point in span)]
    for cx, cy, radius in pads:
        dist = np.hypot(grid_x - cx, grid_y - cy)
        rough *= _smoothstep(radius - 0.2, radius, dist).astype(np.float32)
    wave = np.sin(2.0 * np.pi * grid_x / 0.85) * np.cos(2.0 * np.pi * grid_y / 1.10)
    wave += 0.75 * np.sin(2.0 * np.pi * grid_x / 0.46 + 0.8) * np.sin(2.0 * np.pi * grid_y / 0.62 + 0.5)
    unit = np.clip(0.30 + 0.70 * (0.5 + 0.5 * np.tanh(wave)), 0.0, 1.0)
    height = (amplitude * rough * unit).astype(np.float32)
    return height, (float(x0), float(y0)), float(horizontal_scale), pickups, goals


def paste_tile(height: np.ndarray, tile_index: int, tile: np.ndarray, cols: int, stride: int) -> None:
    """Replace one course inside a bank. Other tiles stay put."""
    column = tile_index * stride
    height[:, column : column + cols] = tile


def sample_height(
    height: np.ndarray,
    origin: tuple[float, float],
    horizontal_scale: float,
    xy: np.ndarray,
) -> np.ndarray:
    """Bilinear height at world xy points shaped (n, 2)."""
    points = np.asarray(xy, dtype=np.float32)
    u = np.clip((points[:, 0] - origin[0]) / horizontal_scale, 0.0, height.shape[0] - 1.001)
    v = np.clip((points[:, 1] - origin[1]) / horizontal_scale, 0.0, height.shape[1] - 1.001)
    i0 = np.floor(u).astype(np.int32)
    j0 = np.floor(v).astype(np.int32)
    du = u - i0
    dv = v - j0
    h00 = height[i0, j0]
    h10 = height[i0 + 1, j0]
    h01 = height[i0, j0 + 1]
    h11 = height[i0 + 1, j0 + 1]
    return ((h00 * (1.0 - du) + h10 * du) * (1.0 - dv) + (h01 * (1.0 - du) + h11 * du) * dv).astype(np.float32)
