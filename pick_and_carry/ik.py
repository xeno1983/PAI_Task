"""Numeric position IK for the Unitree Z1 on the Go2.

FK/IK run in the arm-mount frame (``z1_link00``): the fixed joint
``arm_mount_joint`` places this frame at ``(-0.05, 0, 0.08)`` in the Go2 base.
Only TCP position is solved (orientation free). Genesis-independent torch code.
"""

from __future__ import annotations

import math

import torch

# Fixed mount of the Z1 base on the Go2 torso (URDF ``arm_mount_joint``).
ARM_MOUNT_IN_BASE = (-0.05, 0.0, 0.08)

# Revolute chain after ``z1_link00``: origin (xyz), axis (unit).
_Z1_JOINTS: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, 0.0585), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.045), (0.0, 1.0, 0.0)),
    ((-0.35, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.218, 0.0, 0.057), (0.0, 1.0, 0.0)),
    ((0.07, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0492, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
# After joint6: stator (+0.051) then tcp (+0.07) along parent X.
_TCP_FROM_LINK06 = (0.1212, 0.0, 0.0)

# Carry / walk-with-object hold point in the Go2 base frame (near default EE).
CARRY_POINT_IN_BASE = (0.35, 0.0, 0.48)
PLACE_HEIGHT_OFFSET = 0.12
GRASP_HEIGHT_OFFSET = 0.02


def _device_dtype(q: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    return q.device, q.dtype


def _eye(n: int, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eye = torch.eye(4, device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(batch, 4, 4).clone()


def _rot_axis(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation matrices ``(N, 3, 3)`` for unit axis ``(3,)`` and angle ``(N,)``."""
    n = angle.shape[0]
    device, dtype = angle.device, angle.dtype
    x, y, z = axis.tolist()
    c = torch.cos(angle)
    s = torch.sin(angle)
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    # K cross-product matrix pieces.
    kxx = x * x
    kyy = y * y
    kzz = z * z
    kxy = x * y
    kxz = x * z
    kyz = y * z
    r00 = c + kxx * (one - c)
    r01 = kxy * (one - c) - z * s
    r02 = kxz * (one - c) + y * s
    r10 = kxy * (one - c) + z * s
    r11 = c + kyy * (one - c)
    r12 = kyz * (one - c) - x * s
    r20 = kxz * (one - c) - y * s
    r21 = kyz * (one - c) + x * s
    r22 = c + kzz * (one - c)
    rows = (
        torch.stack((r00, r01, r02), dim=-1),
        torch.stack((r10, r11, r12), dim=-1),
        torch.stack((r20, r21, r22), dim=-1),
    )
    return torch.stack(rows, dim=-2)


def _trans(xyz: tuple[float, float, float], batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = _eye(4, batch, device, dtype)
    t[:, 0, 3] = xyz[0]
    t[:, 1, 3] = xyz[1]
    t[:, 2, 3] = xyz[2]
    return t


def forward_kinematics(q: torch.Tensor) -> torch.Tensor:
    """TCP position in the arm-mount frame. ``q`` is ``(N, 6)`` joint angles."""
    if q.ndim == 1:
        q = q.unsqueeze(0)
    n = q.shape[0]
    device, dtype = _device_dtype(q)
    t = _eye(4, n, device, dtype)
    for i, (origin, axis) in enumerate(_Z1_JOINTS):
        ax = torch.tensor(axis, device=device, dtype=dtype)
        t = t @ _trans(origin, n, device, dtype)
        r = _rot_axis(ax, q[:, i])
        rot = _eye(4, n, device, dtype)
        rot[:, :3, :3] = r
        t = t @ rot
    t = t @ _trans(_TCP_FROM_LINK06, n, device, dtype)
    return t[:, :3, 3]


def numerical_jacobian(q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Position Jacobian ``(N, 3, 6)`` via central differences."""
    if q.ndim == 1:
        q = q.unsqueeze(0)
    n, dof = q.shape
    jac = torch.zeros((n, 3, dof), device=q.device, dtype=q.dtype)
    for j in range(dof):
        dq = torch.zeros_like(q)
        dq[:, j] = eps
        jac[:, :, j] = (forward_kinematics(q + dq) - forward_kinematics(q - dq)) / (2.0 * eps)
    return jac


def solve_ik(
    q0: torch.Tensor,
    target_xyz: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    iterations: int = 40,
    damping: float = 0.08,
    tol: float = 0.01,
    step: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Damped least-squares position IK.

    Returns ``(q_ik, error_norm)`` with shapes ``(N, 6)`` and ``(N,)``.
    """
    if q0.ndim == 1:
        q0 = q0.unsqueeze(0)
    if target_xyz.ndim == 1:
        target_xyz = target_xyz.unsqueeze(0)
    q = q0.clone()
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    for _ in range(iterations):
        pos = forward_kinematics(q)
        err = target_xyz - pos
        err_norm = err.norm(dim=-1)
        if bool((err_norm < tol).all()):
            break
        jac = numerical_jacobian(q)
        # dq = J^T (J J^T + λ² I)^{-1} err
        jjt = jac @ jac.transpose(-1, -2)
        lam2 = (damping ** 2) * eye
        rhs = torch.linalg.solve(jjt + lam2, err.unsqueeze(-1)).squeeze(-1)
        dq = (jac.transpose(-1, -2) @ rhs.unsqueeze(-1)).squeeze(-1)
        q = (q + step * dq).clamp(lower, upper)
    err_norm = (target_xyz - forward_kinematics(q)).norm(dim=-1)
    return q, err_norm


def base_to_mount(points_base: torch.Tensor) -> torch.Tensor:
    """Translate Go2-base points into the arm-mount frame."""
    offset = torch.tensor(ARM_MOUNT_IN_BASE, device=points_base.device, dtype=points_base.dtype)
    return points_base - offset


def mount_to_base(points_mount: torch.Tensor) -> torch.Tensor:
    offset = torch.tensor(ARM_MOUNT_IN_BASE, device=points_mount.device, dtype=points_mount.dtype)
    return points_mount + offset
