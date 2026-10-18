"""Open-loop diagonal trot teacher for Go2 legs.

Produces policy-space actions (12 legs + zeros for arm/gripper) so a behavioral
clone can warm-start PPO. The gait is a simple phase oscillator (CPG-style):
FL+RR share a phase, FR+RL are opposite. Amplitude scales with the forward
speed command; near-zero commands hold the standing pose.
"""

from __future__ import annotations

import math

import torch

from pick_and_carry.config import (
    ACTION_LAYOUT,
    DEFAULT_LEG_ANGLES,
    LEG_JOINTS,
    NUM_ACTIONS,
    layout_slices,
)

# Per-leg joint offsets within the 12-D leg block.
_HIP = 0
_THIGH = 1
_CALF = 2

# Diagonal trot pairs (indices into LEG_JOINTS // 3).
_PAIR_A = (0, 3)  # FL, RR
_PAIR_B = (1, 2)  # FR, RL
# Go2 thigh/calf: increasing thigh from the stand pose drives the base aft in Genesis,
# so the open-loop "forward" swing must be negated for +vx to walk +X.
_SAGITTAL_SIGN = -1.0


class TrotTeacher:
    """Per-env phase oscillator that outputs full 19-D actions (legs only nonzero)."""

    def __init__(
        self,
        num_envs: int,
        *,
        dt: float = 0.02,
        frequency_hz: float = 2.0,
        thigh_amp: float = 0.35,
        calf_amp: float = 0.45,
        hip_amp: float = 0.05,
        leg_action_scale: float = 0.25,
        device: torch.device | None = None,
    ):
        self.dt = float(dt)
        self.base_freq = float(frequency_hz)
        self.thigh_amp = float(thigh_amp)
        self.calf_amp = float(calf_amp)
        self.hip_amp = float(hip_amp)
        self.leg_action_scale = float(leg_action_scale)
        self.device = device or torch.device("cpu")
        self.phase = torch.rand(num_envs, device=self.device) * (2.0 * math.pi)
        self._default = torch.tensor(
            [DEFAULT_LEG_ANGLES[name] for name in LEG_JOINTS],
            dtype=torch.float32,
            device=self.device,
        )
        self._leg_slice = layout_slices(ACTION_LAYOUT)["leg"]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.phase.uniform_(0.0, 2.0 * math.pi)
            return
        if env_ids.numel() == 0:
            return
        self.phase[env_ids] = torch.rand(env_ids.numel(), device=self.device) * (2.0 * math.pi)

    def act(self, command: torch.Tensor) -> torch.Tensor:
        """``command`` is (N, 3) body-frame [vx, vy, yaw]. Returns (N, 19) actions.

        Forward ``vx > 0`` uses the nominal trot. Backward ``vx < 0`` flips the
        thigh/calf swing so the teacher matches the commanded fore–aft direction
        (previously ``|vx|`` drove a forward-only gait and mislabeled reverse).
        Near-zero speed holds the standing pose.
        """
        n = command.shape[0]
        vx = command[:, 0]
        vy = command[:, 1]
        yaw = command[:, 2]
        speed = vx.abs().clamp(0.0, 0.6)
        # Sagittal locomote only when |vx| is clear; pure lateral/yaw stays nearly still.
        sagittal = (speed > 0.08).to(dtype=command.dtype)
        # +1 forward, -1 backward; 0 when standing in x.
        fore_aft = torch.sign(vx)
        fore_aft = torch.where(sagittal > 0, fore_aft, torch.zeros_like(fore_aft))
        amp_scale = sagittal * (0.35 + 0.65 * (speed / 0.50).clamp(0.0, 1.0))
        freq = self.base_freq * (0.85 + 0.30 * (speed / 0.50).clamp(0.0, 1.0))
        # Keep phase ticking only while walking so standstill does not drift the gait clock.
        self.phase = torch.where(
            sagittal > 0,
            (self.phase + 2.0 * math.pi * freq * self.dt) % (2.0 * math.pi),
            self.phase,
        )

        targets = self._default.expand(n, -1).clone()
        for leg_i in range(4):
            if leg_i in _PAIR_A:
                phi = self.phase
            else:
                phi = self.phase + math.pi
            base = 3 * leg_i
            # fore_aft matches command vx; _SAGITTAL_SIGN maps that onto joint angles that
            # actually move the base the same way in simulation.
            swing = torch.sin(phi) * fore_aft * _SAGITTAL_SIGN
            targets[:, base + _THIGH] = self._default[base + _THIGH] + self.thigh_amp * amp_scale * swing
            targets[:, base + _CALF] = self._default[base + _CALF] - self.calf_amp * amp_scale * swing
            # Small hip for lateral / yaw assist (scaled by how much we are locomoting).
            hip = self.hip_amp * amp_scale * (0.5 * vy.sign() * vy.abs().clamp(0.0, 0.2) / 0.2)
            if leg_i in (0, 2):  # left
                targets[:, base + _HIP] = self._default[base + _HIP] + hip - 0.03 * amp_scale * yaw
            else:
                targets[:, base + _HIP] = self._default[base + _HIP] - hip + 0.03 * amp_scale * yaw

        leg_act = (targets - self._default) / max(self.leg_action_scale, 1e-6)
        action = torch.zeros((n, NUM_ACTIONS), dtype=torch.float32, device=self.device)
        action[:, self._leg_slice] = leg_act
        return action
