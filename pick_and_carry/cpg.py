"""Learnable diagonal-trot CPG for Go2 legs.

The policy outputs a compact parameter vector (not 12 joint residuals). The
environment keeps a per-env phase and expands parameters into leg joint
actions in the same units as the old 12-D leg policy space
``(target - default) / leg_action_scale``.

Parameter layout (``CPG_PARAM_DIM``):
  0 freq      – stride frequency
  1 amp       – signed overall amplitude (neg ⇒ reverse calf phase)
  2 thigh     – thigh swing scale
  3 calf      – calf swing scale
  4 lift      – swing-leg clearance
  5 hip_lat   – lateral hip (strafe)
  6 hip_yaw   – yaw hip bias
  7 turn      – left/right stride asymmetry
  8 bias_th   – mean thigh offset (sit / crouch)
  9 bias_cf   – mean calf offset
"""

from __future__ import annotations

import math

import torch

from pick_and_carry.config import CPG_PARAM_DIM, DEFAULT_LEG_ANGLES, LEG_JOINTS

CPG_PARAM_NAMES = (
    "freq",
    "amp",
    "thigh",
    "calf",
    "lift",
    "hip_lat",
    "hip_yaw",
    "turn",
    "bias_th",
    "bias_cf",
)

_HIP = 0
_THIGH = 1
_CALF = 2
_PAIR_A = (0, 3)  # FL, RR
_SAGITTAL_SIGN = 1.0


def _unit(raw: torch.Tensor) -> torch.Tensor:
    return raw.clamp(-1.0, 1.0)


def decode_cpg_params(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map policy outputs ``(N, 10)`` roughly in [-1, 1] to physical CPG gains."""
    u = _unit(raw)
    signed_amp = u[:, 1]
    return {
        "freq_hz": 2.25 + 1.00 * u[:, 0],
        "amp": signed_amp.abs(),
        "going_back": (signed_amp < 0).to(dtype=raw.dtype),
        "thigh_amp": 0.28 * 0.5 * (u[:, 2] + 1.0),
        "calf_amp": 0.45 * 0.5 * (u[:, 3] + 1.0),
        "lift_amp": 0.16 * 0.5 * (u[:, 4] + 1.0),
        "hip_lat": 0.12 * u[:, 5],
        "hip_yaw": 0.22 * u[:, 6],
        "turn": u[:, 7],
        "bias_th": 0.70 * u[:, 8],
        "bias_cf": 0.90 * u[:, 9],
    }


def encode_cpg_params(
    *,
    freq_hz: float | torch.Tensor = 2.5,
    amp: float | torch.Tensor = 0.0,
    reverse: bool | torch.Tensor = False,
    thigh_amp: float | torch.Tensor = 0.22,
    calf_amp: float | torch.Tensor = 0.38,
    lift_amp: float | torch.Tensor = 0.12,
    hip_lat: float | torch.Tensor = 0.0,
    hip_yaw: float | torch.Tensor = 0.0,
    turn: float | torch.Tensor = 0.0,
    bias_th: float | torch.Tensor = 0.0,
    bias_cf: float | torch.Tensor = 0.0,
    n: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Inverse of ``decode_cpg_params`` for teacher / BC targets."""
    device = device or torch.device("cpu")

    def as_col(value: float | torch.Tensor | bool) -> torch.Tensor:
        if torch.is_tensor(value):
            t = value.to(device=device, dtype=torch.float32).reshape(-1)
            if t.numel() == 1 and n > 1:
                t = t.expand(n)
            return t
        return torch.full((n,), float(value), device=device)

    freq_u = ((as_col(freq_hz) - 2.25) / 1.00).clamp(-1.0, 1.0)
    amp_mag = as_col(amp).clamp(0.0, 1.0)
    rev = as_col(reverse) > 0.5
    amp_u = torch.where(rev, -amp_mag, amp_mag)
    thigh_u = (2.0 * as_col(thigh_amp) / 0.28 - 1.0).clamp(-1.0, 1.0)
    calf_u = (2.0 * as_col(calf_amp) / 0.45 - 1.0).clamp(-1.0, 1.0)
    lift_u = (2.0 * as_col(lift_amp) / 0.16 - 1.0).clamp(-1.0, 1.0)
    hip_lat_u = (as_col(hip_lat) / 0.12).clamp(-1.0, 1.0)
    hip_yaw_u = (as_col(hip_yaw) / 0.22).clamp(-1.0, 1.0)
    turn_u = as_col(turn).clamp(-1.0, 1.0)
    bias_th_u = (as_col(bias_th) / 0.70).clamp(-1.0, 1.0)
    bias_cf_u = (as_col(bias_cf) / 0.90).clamp(-1.0, 1.0)
    return torch.stack(
        (freq_u, amp_u, thigh_u, calf_u, lift_u, hip_lat_u, hip_yaw_u, turn_u, bias_th_u, bias_cf_u),
        dim=-1,
    )


class LegCPG:
    """Per-env phase oscillator that maps CPG params → 12-D leg actions."""

    def __init__(
        self,
        num_envs: int,
        *,
        dt: float = 0.02,
        leg_action_scale: float = 0.25,
        device: torch.device | None = None,
    ):
        self.dt = float(dt)
        self.leg_action_scale = float(leg_action_scale)
        self.device = device or torch.device("cpu")
        self.num_envs = int(num_envs)
        self.phase = torch.zeros(num_envs, device=self.device)
        self._default = torch.tensor(
            [DEFAULT_LEG_ANGLES[name] for name in LEG_JOINTS],
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.phase.zero_()
            return
        if env_ids.numel() == 0:
            return
        self.phase[env_ids] = 0.0

    def step(self, raw_params: torch.Tensor) -> torch.Tensor:
        """Advance phase and return leg actions ``(N, 12)`` in policy units."""
        if raw_params.shape[-1] != CPG_PARAM_DIM:
            raise ValueError(f"expected CPG dim {CPG_PARAM_DIM}, got {raw_params.shape[-1]}")
        n = raw_params.shape[0]
        p = decode_cpg_params(raw_params)
        amp = p["amp"]
        moving = (amp > 0.05).to(dtype=raw_params.dtype)
        phase = self.phase[:n]
        phase = torch.where(
            moving > 0,
            (phase + 2.0 * math.pi * p["freq_hz"] * self.dt) % (2.0 * math.pi),
            phase,
        )
        self.phase[:n] = phase

        turn = p["turn"]
        turn_gain = torch.where(turn < 0.0, torch.full_like(turn, 0.55), torch.ones_like(turn))
        turn_eff = turn * turn_gain
        left_stride = (1.0 - 0.30 * turn_eff).clamp(0.75, 1.25)
        right_stride = (1.0 + 0.30 * turn_eff).clamp(0.75, 1.25)
        calf_phase = p["going_back"] * math.pi

        targets = self._default.expand(n, -1).clone()
        targets[:, _THIGH::3] = targets[:, _THIGH::3] + p["bias_th"].unsqueeze(-1)
        targets[:, _CALF::3] = targets[:, _CALF::3] + p["bias_cf"].unsqueeze(-1)

        for leg_i in range(4):
            phi = phase if leg_i in _PAIR_A else phase + math.pi
            base = 3 * leg_i
            is_left = leg_i in (0, 2)
            stride = left_stride if is_left else right_stride
            # Go2: thigh/calf must swing with opposite signs for +vx → +X travel.
            # Matching signs push the base aft (local gait sweeps: calf=+1 → dx<0).
            swing = torch.sin(phi) * _SAGITTAL_SIGN * moving
            calf_swing = torch.sin(phi + calf_phase) * _SAGITTAL_SIGN * moving
            lift = torch.relu(-torch.sin(phi)) * p["lift_amp"] * amp * moving
            targets[:, base + _THIGH] = targets[:, base + _THIGH] + p["thigh_amp"] * amp * stride * swing
            targets[:, base + _CALF] = (
                targets[:, base + _CALF] - p["calf_amp"] * amp * stride * calf_swing - lift
            )
            hip = p["hip_lat"] * amp + p["hip_yaw"]
            if is_left:
                targets[:, base + _HIP] = self._default[base + _HIP] + hip
            else:
                targets[:, base + _HIP] = self._default[base + _HIP] - hip

        leg_act = (targets - self._default) / max(self.leg_action_scale, 1e-6)
        return leg_act.clamp(-4.0, 4.0)


def command_to_cpg_params(
    command: torch.Tensor,
    *,
    base_freq: float = 2.5,
    thigh_amp: float = 0.22,
    calf_amp: float = 0.38,
    lift_amp: float = 0.12,
    hip_amp: float = 0.08,
) -> torch.Tensor:
    """Open-loop teacher mapping body command ``(N,3)`` → CPG params ``(N,10)``."""
    vx = command[:, 0]
    vy = command[:, 1]
    yaw = command[:, 2]
    speed = vx.abs().clamp(0.0, 0.6)
    yaw_mag = yaw.abs().clamp(0.0, 1.2)
    moving = ((speed > 0.05) | (yaw_mag > 0.08) | (vy.abs() > 0.05)).to(dtype=command.dtype)
    going_back = vx < -0.06
    speed_n = (speed / 0.35).clamp(0.0, 1.0)
    yaw_n = (yaw_mag / 0.70).clamp(0.0, 1.0)
    amp = moving * (0.50 + 0.55 * torch.maximum(speed_n, yaw_n)).clamp(0.0, 1.0)
    freq = base_freq * (0.90 + 0.25 * speed_n + 0.20 * yaw_n)
    turn_gain = torch.where(yaw < 0.0, torch.full_like(yaw, 0.55), torch.ones_like(yaw))
    turn = (yaw / 0.70).clamp(-1.0, 1.0)
    hip_lat = hip_amp * amp * (0.5 * vy.sign() * vy.abs().clamp(0.0, 0.25) / 0.25)
    hip_yaw = 0.18 * amp * yaw * turn_gain
    return encode_cpg_params(
        freq_hz=freq,
        amp=amp,
        reverse=going_back,
        thigh_amp=thigh_amp,
        calf_amp=calf_amp,
        lift_amp=lift_amp,
        hip_lat=hip_lat,
        hip_yaw=hip_yaw,
        turn=turn,
        bias_th=0.0,
        bias_cf=0.0,
        n=command.shape[0],
        device=command.device,
    )
