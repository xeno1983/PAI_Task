"""Numeric position IK for the Unitree Z1 on the Go2.

FK/IK run in the arm-mount frame (``z1_link00``): the fixed joint
``arm_mount_joint`` places this frame at ``(-0.05, 0, 0.08)`` in the Go2 base.
Only TCP position is solved (orientation free). Genesis-independent torch code.

Uses an analytic position Jacobian (one FK pass per iteration) — the previous
finite-difference Jacobian needed 12 extra FK calls per iter and dominated
reach/haul/carry step time.
"""

from __future__ import annotations

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

_NUM_JOINTS = len(_Z1_JOINTS)


def _device_dtype(q: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    return q.device, q.dtype


def _rot_axis_xyz(x: float, y: float, z: float, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues ``(N, 3, 3)`` for fixed unit axis ``(x,y,z)`` and angle ``(N,)``."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    one = torch.ones_like(c)
    kxx, kyy, kzz = x * x, y * y, z * z
    kxy, kxz, kyz = x * y, x * z, y * z
    r00 = c + kxx * (one - c)
    r01 = kxy * (one - c) - z * s
    r02 = kxz * (one - c) + y * s
    r10 = kxy * (one - c) + z * s
    r11 = c + kyy * (one - c)
    r12 = kyz * (one - c) - x * s
    r20 = kxz * (one - c) - y * s
    r21 = kyz * (one - c) + x * s
    r22 = c + kzz * (one - c)
    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


def _matmul33(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def _apply_rot_trans(
    rot: torch.Tensor,
    pos: torch.Tensor,
    local_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """``pos + rot @ local_xyz`` for batched ``rot (N,3,3)``, ``pos (N,3)``."""
    ox, oy, oz = local_xyz
    return pos + ox * rot[:, :, 0] + oy * rot[:, :, 1] + oz * rot[:, :, 2]


def forward_kinematics(q: torch.Tensor) -> torch.Tensor:
    """TCP position in the arm-mount frame. ``q`` is ``(N, 6)`` joint angles."""
    pos, _ = forward_kinematics_and_jacobian(q)
    return pos


def forward_kinematics_and_jacobian(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TCP position ``(N, 3)`` and analytic position Jacobian ``(N, 3, 6)``.

    Column ``j`` is ``ω_j × (p_tcp - o_j)`` with axis/origin expressed in the mount frame
    before joint ``j`` rotates.
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
    n = q.shape[0]
    device, dtype = _device_dtype(q)
    # Accumulated rotation / translation of the current link frame.
    rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, 3, 3)
    pos = torch.zeros((n, 3), device=device, dtype=dtype)
    axes = []
    origins = []
    for i, (origin, axis) in enumerate(_Z1_JOINTS):
        pos = _apply_rot_trans(rot, pos, origin)
        ax_x, ax_y, ax_z = axis
        # Joint axis in mount frame (current link orientation).
        axis_w = ax_x * rot[:, :, 0] + ax_y * rot[:, :, 1] + ax_z * rot[:, :, 2]
        axes.append(axis_w)
        origins.append(pos)
        r_local = _rot_axis_xyz(ax_x, ax_y, ax_z, q[:, i])
        rot = _matmul33(rot, r_local)
    pos = _apply_rot_trans(rot, pos, _TCP_FROM_LINK06)
    jac = torch.zeros((n, 3, _NUM_JOINTS), device=device, dtype=dtype)
    for j in range(_NUM_JOINTS):
        jac[:, :, j] = torch.cross(axes[j], pos - origins[j], dim=-1)
    return pos, jac


def numerical_jacobian(q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Position Jacobian ``(N, 3, 6)`` via central differences (tests / reference)."""
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
    iterations: int = 8,
    damping: float = 0.08,
    tol: float = 0.01,
    step: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Damped least-squares position IK.

    Returns ``(q_ik, error_norm)`` with shapes ``(N, 6)`` and ``(N,)``.

    Runs a fixed iteration count (no host sync). Early-exit via ``bool(err < tol)``
    was removed: it stalled reach/haul/catch training (~10 FPS vs ~2000 for walk).
    ``tol`` is kept for API compatibility and is unused.
    """
    _ = tol  # unused: host-side early exit would sync the GPU every step
    if q0.ndim == 1:
        q0 = q0.unsqueeze(0)
    if target_xyz.ndim == 1:
        target_xyz = target_xyz.unsqueeze(0)
    q = q0.clone()
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    pos = q.new_zeros(q.shape[0], 3)
    for _ in range(max(1, int(iterations))):
        pos, jac = forward_kinematics_and_jacobian(q)
        err = target_xyz - pos
        # dq = J^T (J J^T + λ² I)^{-1} err
        jjt = jac @ jac.transpose(-1, -2)
        lam2 = (damping ** 2) * eye
        rhs = torch.linalg.solve(jjt + lam2, err.unsqueeze(-1)).squeeze(-1)
        dq = (jac.transpose(-1, -2) @ rhs.unsqueeze(-1)).squeeze(-1)
        q = (q + step * dq).clamp(lower, upper)
    pos = forward_kinematics(q)
    err_norm = (target_xyz - pos).norm(dim=-1)
    return q, err_norm


def base_to_mount(points_base: torch.Tensor) -> torch.Tensor:
    """Translate Go2-base points into the arm-mount frame."""
    offset = torch.tensor(ARM_MOUNT_IN_BASE, device=points_base.device, dtype=points_base.dtype)
    return points_base - offset


def mount_to_base(points_mount: torch.Tensor) -> torch.Tensor:
    offset = torch.tensor(ARM_MOUNT_IN_BASE, device=points_mount.device, dtype=points_mount.dtype)
    return points_mount + offset
