"""Learnable diagonal-trot CPG for Go2 legs.

The policy outputs a compact parameter vector (not 12 joint residuals). The
environment keeps per-leg phases and expands parameters into leg joint
actions in the same units as the old 12-D leg policy space
``(target - default) / leg_action_scale``.

Parameter layout (``CPG_PARAM_DIM`` = 15):
  0 freq       – stride frequency
  1 amp        – signed overall amplitude (neg ⇒ reverse calf phase)
  2 thigh      – thigh swing scale
  3 calf       – calf swing scale
  4 lift       – swing-leg clearance
  5 hip_lat    – lateral hip (strafe)
  6 hip_yaw    – yaw hip bias
  7 turn       – left/right stride asymmetry
  8 bias_th    – mean thigh offset (sit / crouch)
  9 bias_cf    – mean calf offset
 10 dphi_ab    – pair-B offset from nominal π (uneven terrain)
 11 dphi_a     – within pair A: RR relative to FL
 12 dphi_b     – within pair B: RL relative to FR
 13 couple     – soft diagonal coupling strength
 14 entrain    – contact-triggered phase pull (stance)

Nominal diagonal structure (legs FL, FR, RL, RR = 0..3):
  FL: Φ
  RR: Φ + dphi_a          (pair A)
  FR: Φ + π + dphi_ab
  RL: Φ + π + dphi_ab + dphi_b   (pair B)

Soft coupling pulls each leg phase toward that structure. Contact entrainment
pulls a foot's phase toward early stance (0) on touchdown. Teacher defaults
keep couple≈1 and entrain≈0 so flat-ground BC/GAIL stay rigid diagonal.
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
    "dphi_ab",
    "dphi_a",
    "dphi_b",
    "couple",
    "entrain",
)

_HIP = 0
_THIGH = 1
_CALF = 2
_PAIR_A = (0, 3)  # FL, RR
_PAIR_B = (1, 2)  # FR, RL
# Locked by scripts/diag_trot_direction.py (Genesis, gripper / +X = front):
#   sagittal=+1 and thigh/calf SAME swing sign → dx ≈ +4 m / 4 s (forward)
#   opposite calf sign walked aft. Do not "fix" polarity without re-running that script.
_SAGITTAL_SIGN = 1.0
_CALF_SWING_SIGN = 1.0  # +1 = same as thigh; -1 = opposite (known-bad for +X)

_DPHI_AB_SCALE = 0.70  # rad
_DPHI_WITHIN_SCALE = 0.50  # rad
_COUPLE_MIN = 0.15
_COUPLE_MAX = 1.00
_ENTRAIN_MAX = 0.85  # fraction of wrap(0 - φ) applied on touchdown


def _unit(raw: torch.Tensor) -> torch.Tensor:
    return raw.clamp(-1.0, 1.0)


def _wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to (−π, π]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _angle_mean(angles: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Circular mean of angles (radians)."""
    return torch.atan2(torch.sin(angles).mean(dim=dim), torch.cos(angles).mean(dim=dim))


def decode_cpg_params(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map policy outputs ``(N, 15)`` roughly in [-1, 1] to physical CPG gains."""
    if raw.shape[-1] != CPG_PARAM_DIM:
        raise ValueError(f"expected CPG dim {CPG_PARAM_DIM}, got {raw.shape[-1]}")
    u = _unit(raw)
    signed_amp = u[:, 1]
    couple_u = 0.5 * (u[:, 13] + 1.0)
    entrain_u = 0.5 * (u[:, 14] + 1.0)
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
        "dphi_ab": _DPHI_AB_SCALE * u[:, 10],
        "dphi_a": _DPHI_WITHIN_SCALE * u[:, 11],
        "dphi_b": _DPHI_WITHIN_SCALE * u[:, 12],
        "couple": _COUPLE_MIN + (_COUPLE_MAX - _COUPLE_MIN) * couple_u,
        "entrain": _ENTRAIN_MAX * entrain_u,
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
    dphi_ab: float | torch.Tensor = 0.0,
    dphi_a: float | torch.Tensor = 0.0,
    dphi_b: float | torch.Tensor = 0.0,
    couple: float | torch.Tensor = 1.0,
    entrain: float | torch.Tensor = 0.0,
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
    dphi_ab_u = (as_col(dphi_ab) / _DPHI_AB_SCALE).clamp(-1.0, 1.0)
    dphi_a_u = (as_col(dphi_a) / _DPHI_WITHIN_SCALE).clamp(-1.0, 1.0)
    dphi_b_u = (as_col(dphi_b) / _DPHI_WITHIN_SCALE).clamp(-1.0, 1.0)
    couple_span = max(_COUPLE_MAX - _COUPLE_MIN, 1e-6)
    couple_u01 = ((as_col(couple) - _COUPLE_MIN) / couple_span).clamp(0.0, 1.0)
    couple_u = (2.0 * couple_u01 - 1.0).clamp(-1.0, 1.0)
    entrain_u01 = (as_col(entrain) / max(_ENTRAIN_MAX, 1e-6)).clamp(0.0, 1.0)
    entrain_u = (2.0 * entrain_u01 - 1.0).clamp(-1.0, 1.0)
    return torch.stack(
        (
            freq_u,
            amp_u,
            thigh_u,
            calf_u,
            lift_u,
            hip_lat_u,
            hip_yaw_u,
            turn_u,
            bias_th_u,
            bias_cf_u,
            dphi_ab_u,
            dphi_a_u,
            dphi_b_u,
            couple_u,
            entrain_u,
        ),
        dim=-1,
    )


def nominal_leg_phases(
    carrier: torch.Tensor,
    dphi_ab: torch.Tensor,
    dphi_a: torch.Tensor,
    dphi_b: torch.Tensor,
) -> torch.Tensor:
    """Build (N, 4) nominal phases FL, FR, RL, RR from carrier Φ and offsets."""
    # carrier: (N,)
    fl = carrier
    fr = carrier + math.pi + dphi_ab
    rl = carrier + math.pi + dphi_ab + dphi_b
    rr = carrier + dphi_a
    return torch.stack((fl, fr, rl, rr), dim=-1)


class LegCPG:
    """Per-env multi-leg phase oscillator that maps CPG params → 12-D leg actions."""

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
        # Per-leg phases (FL, FR, RL, RR). ``phase`` property mirrors pair-A mean for tests.
        self.phase_legs = torch.zeros((num_envs, 4), device=self.device)
        self._prev_contact = torch.zeros((num_envs, 4), dtype=torch.bool, device=self.device)
        # After reset, the first contact sample only seeds ``_prev_contact`` (no rising pull).
        # Otherwise stance feet that were already down look like fresh touchdowns.
        self._entrain_ready = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._default = torch.tensor(
            [DEFAULT_LEG_ANGLES[name] for name in LEG_JOINTS],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def phase(self) -> torch.Tensor:
        """Carrier phase ≈ circular mean of pair A (FL, RR). Shape (N,)."""
        return _angle_mean(self.phase_legs[:, list(_PAIR_A)], dim=-1)

    @phase.setter
    def phase(self, value: torch.Tensor) -> None:
        """Set all legs to a rigid diagonal around ``value`` (broadcastable (N,) or scalar)."""
        carrier = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if carrier.ndim == 0:
            carrier = carrier.expand(self.num_envs)
        elif carrier.shape[0] != self.num_envs:
            carrier = carrier.reshape(-1)
            if carrier.numel() == 1:
                carrier = carrier.expand(self.num_envs)
        zeros = torch.zeros_like(carrier)
        self.phase_legs = nominal_leg_phases(carrier, zeros, zeros, zeros) % (2.0 * math.pi)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.phase_legs.zero_()
            self._prev_contact.zero_()
            self._entrain_ready.zero_()
            return
        if env_ids.numel() == 0:
            return
        self.phase_legs[env_ids] = 0.0
        self._prev_contact[env_ids] = False
        self._entrain_ready[env_ids] = False

    def step(
        self,
        raw_params: torch.Tensor,
        foot_contact: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance per-leg phases and return leg actions ``(N, 12)`` in policy units.

        ``foot_contact`` is optional ``(N, 4)`` in FL/FR/RL/RR order (truthy = in contact).
        When omitted, entrainment is skipped (unit tests / open-loop).
        The first contact sample after ``reset`` only seeds edge history; rising-edge
        pulls start on the following step.
        """
        if raw_params.shape[-1] != CPG_PARAM_DIM:
            raise ValueError(f"expected CPG dim {CPG_PARAM_DIM}, got {raw_params.shape[-1]}")
        n = raw_params.shape[0]
        p = decode_cpg_params(raw_params)
        amp = p["amp"]
        moving = (amp > 0.05).to(dtype=raw_params.dtype)
        legs = self.phase_legs[:n]

        # 1) Free-run advance (same frequency on all legs).
        delta = 2.0 * math.pi * p["freq_hz"] * self.dt
        legs = torch.where(
            (moving > 0).unsqueeze(-1),
            (legs + delta.unsqueeze(-1)) % (2.0 * math.pi),
            legs,
        )

        # 2) Soft coupling toward diagonal structure with learnable offsets.
        carrier = _angle_mean(legs[:, list(_PAIR_A)], dim=-1)
        target = nominal_leg_phases(carrier, p["dphi_ab"], p["dphi_a"], p["dphi_b"])
        err = _wrap_pi(target - legs)
        couple = p["couple"].unsqueeze(-1) * moving.unsqueeze(-1)
        legs = (legs + couple * err) % (2.0 * math.pi)

        # 3) Contact entrainment: on touchdown, pull phase toward early stance (0).
        if foot_contact is not None:
            contact = foot_contact[:n].to(dtype=torch.bool)
            if contact.shape[-1] != 4:
                raise ValueError(f"foot_contact last dim must be 4, got {contact.shape[-1]}")
            ready = self._entrain_ready[:n].unsqueeze(-1)
            rising = contact & ~self._prev_contact[:n] & ready
            pull = _wrap_pi(-legs)  # toward 0
            gain = p["entrain"].unsqueeze(-1) * rising.to(dtype=raw_params.dtype)
            legs = (legs + gain * pull) % (2.0 * math.pi)
            self._prev_contact[:n] = contact
            self._entrain_ready[:n] = True
        self.phase_legs[:n] = legs

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
            phi = legs[:, leg_i]
            base = 3 * leg_i
            is_left = leg_i in (0, 2)
            stride = left_stride if is_left else right_stride
            # Fore-aft polarity: _SAGITTAL_SIGN and _CALF_SWING_SIGN (Genesis-locked).
            swing = torch.sin(phi) * _SAGITTAL_SIGN * moving
            calf_swing = torch.sin(phi + calf_phase) * _SAGITTAL_SIGN * moving
            lift = torch.relu(-torch.sin(phi)) * p["lift_amp"] * amp * moving
            targets[:, base + _THIGH] = targets[:, base + _THIGH] + p["thigh_amp"] * amp * stride * swing
            targets[:, base + _CALF] = (
                targets[:, base + _CALF]
                + _CALF_SWING_SIGN * p["calf_amp"] * amp * stride * calf_swing
                - lift
            )
            hip = p["hip_lat"] * amp + p["hip_yaw"]
            if is_left:
                targets[:, base + _HIP] = self._default[base + _HIP] + hip
            else:
                targets[:, base + _HIP] = self._default[base + _HIP] - hip

        leg_act = (targets - self._default) / max(self.leg_action_scale, 1e-6)
        return leg_act.clamp(-4.0, 4.0)


# Match teacher / env still gates so BC stand-sit-jump-stop stay amp-free.
_COMMAND_SPEED_EPS = 0.05
_COMMAND_YAW_EPS = 0.08
_COMMAND_STRAFE_EPS = 0.05


def command_is_moving(command: torch.Tensor) -> torch.Tensor:
    """Body-frame command ``(N,3)`` → bool mask: nonzero walk / turn / strafe."""
    speed = command[:, 0].abs()
    strafe = command[:, 1].abs()
    yaw = command[:, 2].abs()
    return (speed > _COMMAND_SPEED_EPS) | (yaw > _COMMAND_YAW_EPS) | (strafe > _COMMAND_STRAFE_EPS)


def gate_cpg_amp_by_command(cpg_params: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """Zero the signed amp channel when the velocity command is still.

    Stand / sit / jump / stop all use ``command≈0``. Without this gate, a BC
    policy that leaks a small amp (regression toward walk demos) starts
    stepping and then self-amplifies through ``last_leg_action`` / base velocity.
    Sit/jump pose biases are left intact.
    """
    if cpg_params.shape[-1] < 2:
        return cpg_params
    moving = command_is_moving(command).to(dtype=cpg_params.dtype)
    out = cpg_params.clone()
    out[:, 1] = out[:, 1] * moving
    return out


def command_to_cpg_params(
    command: torch.Tensor,
    *,
    base_freq: float = 2.5,
    thigh_amp: float = 0.22,
    calf_amp: float = 0.38,
    lift_amp: float = 0.12,
    hip_amp: float = 0.08,
) -> torch.Tensor:
    """Open-loop teacher mapping body command ``(N,3)`` → CPG params ``(N,15)``.

    Phase-offset / entrainment channels stay at rigid-diagonal defaults so the
    teacher remains a classic diagonal trot on flat ground.
    """
    vx = command[:, 0]
    vy = command[:, 1]
    yaw = command[:, 2]
    speed = vx.abs().clamp(0.0, 0.6)
    yaw_mag = yaw.abs().clamp(0.0, 1.2)
    moving = command_is_moving(command).to(dtype=command.dtype)
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
        dphi_ab=0.0,
        dphi_a=0.0,
        dphi_b=0.0,
        couple=1.0,
        entrain=0.0,
        n=command.shape[0],
        device=command.device,
    )
