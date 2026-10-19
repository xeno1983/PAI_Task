"""Open-loop CPG teacher for Go2 (stand / sit / jump / walk / turn / stop).

Produces policy-space actions (CPG params + zeros for arm/gripper) so behavioral
cloning and GAIL warm-start the same action space as PPO. Leg joints are expanded
inside the environment by ``LegCPG``.

Convention: gripper / FL–FR side is front (+X). ``vx > 0`` must walk toward +X.
Sagittal polarity is Genesis-locked in ``cpg._SAGITTAL_SIGN`` / ``cpg._CALF_SWING_SIGN``
(both +1, thigh and calf same swing sign). Re-check with
``scripts/diag_trot_direction.py`` after any swing-sign edit.
"""

from __future__ import annotations

from typing import Sequence

import torch

from pick_and_carry.config import ACTION_LAYOUT, NUM_ACTIONS, layout_slices
from pick_and_carry.cpg import command_to_cpg_params, encode_cpg_params

# Named skills used by demos and BC command sampling.
DEMO_SKILLS: tuple[str, ...] = (
    "stand",
    "sit",
    "jump",
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "stop",
)

# Still skills share command≈0; keep them well represented so BC does not
# regress amp toward the walk majority when standing / after a jump.
BC_SKILL_WEIGHTS: dict[str, float] = {
    "stand": 0.14,
    "sit": 0.10,
    "jump": 0.10,
    "forward": 0.20,
    "backward": 0.12,
    "turn_left": 0.12,
    "turn_right": 0.12,
    "stop": 0.10,
}

DEMO_SKILL_SECONDS: dict[str, float] = {
    "stand": 3.0,
    "sit": 4.0,
    "jump": 5.0,
    "forward": 6.0,
    "backward": 6.0,
    "turn_left": 8.0,
    "turn_right": 8.0,
    "stop": 3.0,
}

# Sit / jump as mean-pose biases relative to the default stand (radians).
_SIT_BIAS_TH = 1.35 - 0.8
_SIT_BIAS_CF = -2.25 - (-1.5)
_JUMP_CROUCH_TH = 1.25 - 0.8
_JUMP_CROUCH_CF = -2.15 - (-1.5)
_JUMP_EXTEND_TH = 0.45 - 0.8
_JUMP_EXTEND_CF = -1.15 - (-1.5)


def skill_command(skill: str, *, device: torch.device | None = None, n: int = 1) -> torch.Tensor:
    """Body-frame [vx, vy, yaw] that matches ``skill`` for demos / BC."""
    device = device or torch.device("cpu")
    table = {
        "stand": (0.0, 0.0, 0.0),
        "sit": (0.0, 0.0, 0.0),
        "jump": (0.0, 0.0, 0.0),
        "forward": (0.35, 0.0, 0.0),
        "backward": (-0.30, 0.0, 0.0),
        "turn_left": (0.22, 0.0, 0.70),
        "turn_right": (0.24, 0.0, -0.55),
        "stop": (0.0, 0.0, 0.0),
    }
    if skill not in table:
        raise ValueError(f"unknown skill {skill!r}; expected one of {DEMO_SKILLS}")
    row = table[skill]
    return torch.tensor([row] * n, dtype=torch.float32, device=device)


class TrotTeacher:
    """Per-env teacher that outputs full CPG+arm actions (CPG only nonzero)."""

    def __init__(
        self,
        num_envs: int,
        *,
        dt: float = 0.02,
        frequency_hz: float = 2.5,
        thigh_amp: float = 0.22,
        calf_amp: float = 0.38,
        lift_amp: float = 0.12,
        hip_amp: float = 0.08,
        leg_action_scale: float = 0.25,
        smooth: float = 0.35,
        device: torch.device | None = None,
    ):
        self.dt = float(dt)
        self.base_freq = float(frequency_hz)
        self.thigh_amp = float(thigh_amp)
        self.calf_amp = float(calf_amp)
        self.lift_amp = float(lift_amp)
        self.hip_amp = float(hip_amp)
        self.leg_action_scale = float(leg_action_scale)
        self.smooth = float(smooth)
        self.device = device or torch.device("cpu")
        self.num_envs = int(num_envs)
        self.skill_id = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.jump_t = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._cpg_slice = layout_slices(ACTION_LAYOUT)["cpg"]
        cpg_dim = self._cpg_slice.stop - self._cpg_slice.start
        self._prev_params = torch.zeros((num_envs, cpg_dim), dtype=torch.float32, device=self.device)
        self._blend = torch.ones(num_envs, dtype=torch.float32, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.jump_t.zero_()
            self._prev_params.zero_()
            self._blend.fill_(1.0)
            return
        if env_ids.numel() == 0:
            return
        self.jump_t[env_ids] = 0.0
        self._prev_params[env_ids] = 0.0
        self._blend[env_ids] = 1.0

    def set_skill(self, skill: str, env_ids: torch.Tensor | None = None) -> None:
        if skill not in DEMO_SKILLS:
            raise ValueError(f"unknown skill {skill!r}; expected one of {DEMO_SKILLS}")
        sid = DEMO_SKILLS.index(skill)
        if env_ids is None:
            entering = bool((self.skill_id != sid).any())
            self.skill_id.fill_(sid)
            if entering:
                self._blend.fill_(0.0)
                if skill == "jump":
                    self.jump_t.zero_()
            return
        prev = self.skill_id[env_ids].clone()
        self.skill_id[env_ids] = sid
        entering = prev != sid
        self._blend[env_ids] = torch.where(entering, torch.zeros_like(self._blend[env_ids]), self._blend[env_ids])
        if skill == "jump":
            self.jump_t[env_ids] = torch.where(entering, torch.zeros_like(self.jump_t[env_ids]), self.jump_t[env_ids])

    def clear_skill(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.skill_id.fill_(-1)
            return
        self.skill_id[env_ids] = -1

    def sample_skills(self, skills: Sequence[str] | None = None) -> None:
        pool = list(skills) if skills is not None else list(DEMO_SKILLS)
        weights = torch.tensor([BC_SKILL_WEIGHTS.get(name, 1.0) for name in pool], device=self.device)
        weights = weights / weights.sum().clamp(min=1e-6)
        ids = torch.multinomial(weights.expand(self.num_envs, -1), 1).squeeze(-1)
        mapped = torch.tensor([DEMO_SKILLS.index(name) for name in pool], device=self.device)
        self.skill_id = mapped[ids]
        self._blend.zero_()
        jumping = self.skill_id == DEMO_SKILLS.index("jump")
        self.jump_t[jumping] = 0.0

    def commands_for_skills(self) -> torch.Tensor:
        out = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        for i, name in enumerate(DEMO_SKILLS):
            mask = self.skill_id == i
            if bool(mask.any()):
                out[mask] = skill_command(name, device=self.device, n=1).expand(int(mask.sum()), -1)
        return out

    def act(
        self,
        command: torch.Tensor,
        skill: str | None = None,
        *,
        advance: bool = True,
    ) -> torch.Tensor:
        """``command`` is (N, 3). Returns (N, NUM_ACTIONS) with CPG params in the cpg slice.

        ``advance=False`` peeks the action without updating blend / phase / jump timers
        (needed for DAgger when the env may apply the student action instead).
        """
        n = command.shape[0]
        if skill is not None:
            self.set_skill(skill)

        params = torch.zeros((n, self._prev_params.shape[-1]), dtype=torch.float32, device=self.device)
        free = self.skill_id < 0
        for i, name in enumerate(DEMO_SKILLS):
            mask = self.skill_id == i
            if not bool(mask.any()):
                continue
            params[mask] = self._params_for_skill(name, command, mask, advance=advance)

        if bool(free.any()):
            params[free] = command_to_cpg_params(
                command[free],
                base_freq=self.base_freq,
                thigh_amp=self.thigh_amp,
                calf_amp=self.calf_amp,
                lift_amp=self.lift_amp,
                hip_amp=self.hip_amp,
            )

        blend_next = (self._blend + self.dt / 0.35).clamp(0.0, 1.0)
        alpha = blend_next.unsqueeze(-1)
        params = alpha * params + (1.0 - alpha) * self._prev_params[:n]
        if self.smooth > 0.0:
            params = (1.0 - self.smooth) * self._prev_params[:n] + self.smooth * params
        if advance:
            self._blend = blend_next
            self._prev_params[:n] = params

        action = torch.zeros((n, NUM_ACTIONS), dtype=torch.float32, device=self.device)
        action[:, self._cpg_slice] = params.clamp(-4.0, 4.0)
        return action

    def snapshot_state(self) -> dict[str, torch.Tensor]:
        return {
            "blend": self._blend.clone(),
            "prev_params": self._prev_params.clone(),
            "jump_t": self.jump_t.clone(),
        }

    def restore_state(self, snap: dict[str, torch.Tensor], env_mask: torch.Tensor | None = None) -> None:
        """Restore CPG phase state. If ``env_mask`` is set, only those envs are restored."""
        if env_mask is None:
            self._blend.copy_(snap["blend"])
            self._prev_params.copy_(snap["prev_params"])
            self.jump_t.copy_(snap["jump_t"])
            return
        sel = env_mask.to(dtype=torch.bool, device=self.device)
        self._blend = torch.where(sel, snap["blend"], self._blend)
        self._prev_params = torch.where(sel.unsqueeze(-1), snap["prev_params"], self._prev_params)
        self.jump_t = torch.where(sel, snap["jump_t"], self.jump_t)

    def _params_for_skill(
        self,
        skill: str,
        command: torch.Tensor,
        mask: torch.Tensor,
        *,
        advance: bool = True,
    ) -> torch.Tensor:
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        n = idx.numel()
        if skill in ("stand", "stop"):
            return encode_cpg_params(amp=0.0, n=n, device=self.device)
        if skill == "sit":
            return encode_cpg_params(
                amp=0.0,
                bias_th=_SIT_BIAS_TH,
                bias_cf=_SIT_BIAS_CF,
                n=n,
                device=self.device,
            )
        if skill == "jump":
            return self._jump_params(idx, advance=advance)
        return command_to_cpg_params(
            command[idx],
            base_freq=self.base_freq,
            thigh_amp=self.thigh_amp,
            calf_amp=self.calf_amp,
            lift_amp=self.lift_amp,
            hip_amp=self.hip_amp,
        )

    def _jump_params(self, idx: torch.Tensor, *, advance: bool = True) -> torch.Tensor:
        if advance:
            self.jump_t[idx] = self.jump_t[idx] + self.dt
        t = self.jump_t[idx] + (0.0 if advance else self.dt)
        n = idx.numel()
        bias_th = torch.zeros(n, device=self.device)
        bias_cf = torch.zeros(n, device=self.device)
        crouch = t < 0.40
        extend = (t >= 0.40) & (t < 0.65)
        bias_th = torch.where(crouch, torch.full_like(bias_th, _JUMP_CROUCH_TH), bias_th)
        bias_cf = torch.where(crouch, torch.full_like(bias_cf, _JUMP_CROUCH_CF), bias_cf)
        bias_th = torch.where(extend, torch.full_like(bias_th, _JUMP_EXTEND_TH), bias_th)
        bias_cf = torch.where(extend, torch.full_like(bias_cf, _JUMP_EXTEND_CF), bias_cf)
        return encode_cpg_params(
            amp=0.0,
            bias_th=bias_th,
            bias_cf=bias_cf,
            n=n,
            device=self.device,
        )
