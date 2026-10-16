"""Geometry, grasp checks, and the staged pick-and-carry reward.

These functions stay independent of Genesis so they can be unit-tested and
reused by the environment.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pick_and_carry.config import RewardScales


def fleet_camera_height(offsets_xy: np.ndarray, spacing: float, fov_deg: float = 50.0) -> float:
    """Height above the block that fits it in a vertical field of view.

    Genesis cameras are created with a far plane of 20 m. A 6 by 6 block at
    the default 3 m spacing already needs to sit higher than that, and anything
    past the far plane is dropped, leaving the background color only.
    """
    if len(offsets_xy) == 0:
        return 8.0
    span = float(np.ptp(np.asarray(offsets_xy, dtype=np.float64), axis=0).max()) + float(spacing)
    half = math.radians(fov_deg * 0.5)
    return 0.5 * span / math.tan(half) * 1.15


def fleet_view_indices(num_envs: int, side: int = 6) -> list[int]:
    """Corner block of the environment grid, at most ``side`` by ``side``.

    Genesis lays environments out with ``ceil(sqrt(n))`` columns. The first
    ``side`` columns of the first ``side`` rows stay in frame. The rest are simulated
    but not drawn.
    """
    if num_envs < 1:
        return []
    per_row = math.ceil(math.sqrt(num_envs))
    span = min(side, per_row, num_envs)
    indices: list[int] = []
    for row in range(span):
        for col in range(span):
            index = row * per_row + col
            if index < num_envs:
                indices.append(index)
    return indices


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion in wxyz order."""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by unit quaternions ``q`` (wxyz)."""
    q_vec = q[..., 1:]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def world_to_base(points_world: torch.Tensor, base_pos: torch.Tensor, base_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Express world points in the robot base frame."""
    return quat_apply(quat_conjugate(base_quat_wxyz), points_world - base_pos)


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw-only quaternion, wxyz."""
    half = 0.5 * yaw
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def pairwise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a - b, dim=-1)


def gripper_is_closed(gripper_opening: torch.Tensor, closed_threshold: float) -> torch.Tensor:
    return gripper_opening < closed_threshold


def grasp_mask(
    ee_pos: torch.Tensor,
    obj_pos: torch.Tensor,
    gripper_opening: torch.Tensor,
    distance_threshold: float,
    closed_threshold: float,
) -> torch.Tensor:
    near = pairwise_distance(ee_pos, obj_pos) < distance_threshold
    return near & gripper_is_closed(gripper_opening, closed_threshold)


def place_mask(
    obj_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    grasped: torch.Tensor,
    was_grasped: torch.Tensor,
    goal_radius: float,
    max_height: float,
) -> torch.Tensor:
    dist_xy = torch.linalg.norm(obj_pos[..., :2] - goal_pos[..., :2], dim=-1)
    on_ground = obj_pos[..., 2] < max_height
    return (~grasped) & was_grasped & (dist_xy < goal_radius) & on_ground


# 0 cross rough ground to the object, 1 pick it up, 2 cross rough ground to the goal, 3 release.
PHASE_WALK_OBJECT = 0
PHASE_PICKUP = 1
PHASE_WALK_GOAL = 2
PHASE_RELEASE = 3


def task_phase(
    grasped: torch.Tensor,
    base_object_xy: torch.Tensor,
    object_goal_xy: torch.Tensor,
    pickup_distance: float,
    release_distance: float,
) -> torch.Tensor:
    """Which part of the rough-terrain course is active: travel, pick up, travel, release."""
    walk_object = (~grasped) & (base_object_xy > pickup_distance)
    pickup = (~grasped) & ~walk_object
    release = grasped & (object_goal_xy <= release_distance)
    walk_goal = grasped & ~release
    phase = torch.zeros(grasped.shape, dtype=torch.long, device=grasped.device)
    phase = torch.where(pickup, torch.full_like(phase, PHASE_PICKUP), phase)
    phase = torch.where(walk_goal, torch.full_like(phase, PHASE_WALK_GOAL), phase)
    phase = torch.where(release, torch.full_like(phase, PHASE_RELEASE), phase)
    return phase


def fallen_mask(projected_gravity: torch.Tensor, base_height: torch.Tensor, max_tilt_deg: float, min_base_height: float) -> torch.Tensor:
    """``projected_gravity`` is gravity expressed in the base frame. Upright is (0, 0, -1)."""
    alignment = -projected_gravity[..., 2]
    tilted = alignment < math.cos(math.radians(max_tilt_deg))
    too_low = base_height < min_base_height
    return tilted | too_low


def compute_stage_reward(
    *,
    ee_pos: torch.Tensor,
    obj_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    base_pos: torch.Tensor,
    grasped: torch.Tensor,
    newly_grasped: torch.Tensor,
    placed: torch.Tensor,
    fallen: torch.Tensor,
    gripper_opening: torch.Tensor,
    gripper_open_width: float,
    actions: torch.Tensor,
    scales: RewardScales,
    dt: float,
    approach_sigma: float,
    transport_sigma: float,
    base_sigma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Staged reward: approach, grasp, carry, place, plus fall and energy terms.

    Dense terms are scaled by ``dt``. Sparse grasp and place bonuses are added
    once per event so they stay visible next to the per-step terms.
    """
    dist_ee = pairwise_distance(ee_pos, obj_pos)
    dist_goal = pairwise_distance(obj_pos, goal_pos)
    dist_base = torch.linalg.norm(base_pos[..., :2] - obj_pos[..., :2], dim=-1)
    free = (~grasped).to(dtype=ee_pos.dtype)
    held = grasped.to(dtype=ee_pos.dtype)
    close_frac = (1.0 - (gripper_opening / gripper_open_width)).clamp(0.0, 1.0)
    near_ee = torch.exp(-dist_ee / approach_sigma)

    terms = {
        "approach_ee": scales.approach_ee * dt * near_ee * free,
        "approach_base": scales.approach_base * dt * torch.exp(-dist_base / base_sigma) * free,
        "grasp_shape": scales.grasp_shape * dt * close_frac * near_ee * free,
        "grasp_bonus": scales.grasp_bonus * newly_grasped.to(dtype=ee_pos.dtype),
        "transport": scales.transport * dt * torch.exp(-dist_goal / transport_sigma) * held,
        "place": scales.place_bonus * placed.to(dtype=ee_pos.dtype),
        "fall": -scales.fall * fallen.to(dtype=ee_pos.dtype),
        "energy": -scales.energy * dt * torch.sum(actions * actions, dim=-1),
        "alive": torch.full((ee_pos.shape[0],), scales.alive * dt, device=ee_pos.device, dtype=ee_pos.dtype),
    }
    reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    return reward, terms


_OPENGL_TO_OPENCV = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def detect_wrist_target(
    depth,
    segmentation,
    target_ids,
    focal: float,
    cx: float,
    cy: float,
    transform,
    near: float = 0.02,
    far: float = 2.0,
    min_pixels: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid of one segmented object in a wrist-camera depth image.

    ``depth`` and ``segmentation`` are (H, W) or (N, H, W). ``transform`` is the
    camera pose, (4, 4) or (N, 4, 4), in the same convention as Genesis.
    Returns world positions (N, 3) and a visibility flag (N,).
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    seg_arr = np.asarray(segmentation)
    if seg_arr.ndim == depth_arr.ndim + 1:
        seg_arr = seg_arr[..., 0]
    if depth_arr.ndim == 2:
        depth_arr = depth_arr[None]
        seg_arr = seg_arr[None]
    count, height, width = depth_arr.shape
    ids = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    if ids.shape[0] == 1 and count > 1:
        ids = np.repeat(ids, count)
    pose = np.asarray(transform, dtype=np.float32)
    if pose.ndim == 2:
        pose = np.repeat(pose[None], count, axis=0)
    cam_pose = pose @ _OPENGL_TO_OPENCV
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    positions = np.zeros((count, 3), dtype=np.float32)
    visible = np.zeros((count,), dtype=bool)
    for index in range(count):
        mask = (seg_arr[index] == ids[index]) & (depth_arr[index] > near) & (depth_arr[index] < far * 0.999)
        if int(mask.sum()) < min_pixels:
            continue
        z = depth_arr[index][mask]
        x = z * (cols[mask].astype(np.float32) + 0.5 - cx) / focal
        y = z * (rows[mask].astype(np.float32) + 0.5 - cy) / focal
        points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
        positions[index] = (points @ cam_pose[index].T)[:, :3].mean(axis=0)
        visible[index] = True
    return positions, visible
