"""Learning algorithms on top of the shared ActorCritic.

``ppo`` – clipped PPO with GAE (the default).
``grpo`` – Group Relative Policy Optimization: critic-free Monte Carlo returns,
normalized within groups of parallel environments at each time step.
``sym_ppo`` – PPO with left–right (sagittal) augmentation on moving-command samples.
"""

from __future__ import annotations

import torch

ALGORITHMS = ("ppo", "grpo", "sym_ppo")
ALGO_LABELS = {
    "ppo": "PPO",
    "grpo": "GRPO",
    "sym_ppo": "Symmetric PPO",
}

# Z1 revolute axes that flip under a sagittal (left–right) reflection: yaw and roll.
_ARM_SIGN_FLIP = (0, 4, 5)  # joint1 (z), joint5 (z), joint6 (x)


def normalize_algo(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {"symmetric_ppo": "sym_ppo", "symmetric": "sym_ppo", "symm_ppo": "sym_ppo"}
    key = aliases.get(key, key)
    if key not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}; choose one of {ALGORITHMS}")
    return key


def monte_carlo_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted return-to-go with no value bootstrap. ``rewards`` and ``dones`` are ``(T, N)``."""
    steps = rewards.shape[0]
    out = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(steps)):
        running = rewards[t] + gamma * (1.0 - dones[t]) * running
        out[t] = running
    return out


def group_relative_advantages(returns: torch.Tensor, group_size: int) -> torch.Tensor:
    """DeepSeek-style group relative advantages on a flat return vector.

    ``returns`` is shaped ``(N,)``. Rows are split into groups of ``group_size``
    (the last incomplete group uses the batch mean). Each group is mean/std normalized.
    """
    flat = returns.reshape(-1)
    n = flat.numel()
    size = max(1, int(group_size))
    if n < size:
        return (flat - flat.mean()) / (flat.std(unbiased=False) + 1e-8)
    usable = (n // size) * size
    head = flat[:usable].view(-1, size)
    mean = head.mean(dim=-1, keepdim=True)
    std = head.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
    normed = (head - mean) / std
    out = flat.clone()
    out[:usable] = normed.reshape(-1)
    if usable < n:
        tail = flat[usable:]
        out[usable:] = (tail - flat.mean()) / (flat.std(unbiased=False) + 1e-8)
    return out


def _swap_leg_blocks(tensor: torch.Tensor, start: int) -> None:
    """In-place: swap FL↔FR and RL↔RR, and negate each hip (lateral) angle."""
    # Layout inside 12: FL(0:3), FR(3:6), RL(6:9), RR(9:12)
    left_front = tensor[:, start : start + 3].clone()
    right_front = tensor[:, start + 3 : start + 6].clone()
    left_rear = tensor[:, start + 6 : start + 9].clone()
    right_rear = tensor[:, start + 9 : start + 12].clone()
    left_front[:, 0] *= -1.0
    right_front[:, 0] *= -1.0
    left_rear[:, 0] *= -1.0
    right_rear[:, 0] *= -1.0
    tensor[:, start : start + 3] = right_front
    tensor[:, start + 3 : start + 6] = left_front
    tensor[:, start + 6 : start + 9] = right_rear
    tensor[:, start + 9 : start + 12] = left_rear


def _flip_arm_signs(tensor: torch.Tensor, start: int) -> None:
    """Negate Z1 joints whose axes reverse under a sagittal reflection."""
    for offset in _ARM_SIGN_FLIP:
        tensor[:, start + offset] *= -1.0


def mirror_observation(obs: torch.Tensor, *, command_has_yaw: bool = True) -> torch.Tensor:
    """Sagittal reflection of an observation batch ``(N, 81)``.

    ``command_has_yaw`` is True for walk commands ``[vx, vy, yaw]``. Catch stores a
    predicted ball position ``[x, y, z]`` in the same slot, so only ``y`` flips then.
    """
    if obs.shape[-1] < 81:
        raise ValueError(f"expected at least 81-dim obs, got {tuple(obs.shape)}")
    out = obs.clone()
    _swap_leg_blocks(out, 0)  # leg_pos
    _swap_leg_blocks(out, 12)  # leg_vel
    out[:, 25] *= -1.0  # projected_gravity y
    out[:, 28] *= -1.0  # base_lin_vel y
    out[:, 30] *= -1.0  # base_ang_vel x
    out[:, 32] *= -1.0  # base_ang_vel z
    # foot_contact FL FR RL RR
    contact = out[:, 34:38].clone()
    out[:, 34] = contact[:, 1]
    out[:, 35] = contact[:, 0]
    out[:, 36] = contact[:, 3]
    out[:, 37] = contact[:, 2]
    _flip_arm_signs(out, 38)  # arm_pos
    _flip_arm_signs(out, 44)  # arm_vel
    out[:, 53] *= -1.0  # ee y
    out[:, 56] *= -1.0  # object y
    out[:, 59] *= -1.0  # goal y (or ball vy in catch)
    out[:, 67] *= -1.0  # command y (vy or predicted py)
    if command_has_yaw:
        out[:, 68] *= -1.0  # command yaw; leave z alone when the slot is a position
    _swap_leg_blocks(out, 69)  # last_leg_action
    return out


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    """Sagittal reflection of a CPG+arm action batch ``(N, CPG+7)``."""
    from pick_and_carry.config import ACTION_LAYOUT, CPG_PARAM_DIM, NUM_ACTIONS, layout_slices

    if action.shape[-1] < NUM_ACTIONS:
        raise ValueError(f"expected at least {NUM_ACTIONS}-dim action, got {tuple(action.shape)}")
    out = action.clone()
    # CPG: flip lateral / yaw / turn channels (indices 5, 6, 7).
    out[:, 5] *= -1.0
    out[:, 6] *= -1.0
    out[:, 7] *= -1.0
    # Phase offsets: L–R swaps pair A↔B within-offsets; negate pair-B π offset.
    if CPG_PARAM_DIM >= 15:
        out[:, 10] *= -1.0  # dphi_ab
        a, b = out[:, 11].clone(), out[:, 12].clone()
        out[:, 11] = b  # dphi_a ↔ dphi_b
        out[:, 12] = a
        # couple (13) and entrain (14) are scalars — unchanged
    arm = layout_slices(ACTION_LAYOUT)["arm"]
    _flip_arm_signs(out, arm.start)
    return out


def moving_command_mask(obs: torch.Tensor, *, command_has_yaw: bool = True) -> torch.Tensor:
    """True where the observation command asks to move (walk) or always True (non-walk slots).

    Walk commands are ``[vx, vy, yaw]``. Still-stand samples are excluded from Sym
    mirroring so left–right augmentation does not invent fake symmetries while stopped
    (and under asymmetric arm disturb). Catch stores a predicted ball pose in the same
    slot — those batches keep full mirroring.
    """
    from pick_and_carry.config import OBS_LAYOUT, layout_slices
    from pick_and_carry.curriculum import MOVE_LIN, MOVE_YAW

    if not command_has_yaw:
        return torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
    cmd = obs[:, layout_slices(OBS_LAYOUT)["command"]]
    lin = cmd[:, :2].norm(dim=-1) > float(MOVE_LIN)
    yaw = cmd[:, 2].abs() > float(MOVE_YAW)
    return lin | yaw


def augment_symmetric(
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    old_values: torch.Tensor | None = None,
    *,
    agent=None,
    command_has_yaw: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Augment a PPO batch with left–right mirrors of *moving-command* samples.

    Still-stand walk commands are left unaugmented. When ``agent`` is given,
    ``old_log_prob`` for the mirrored half is recomputed as ``log π(g·a | g·s)``
    under the behavior policy, so the PPO ratio stays valid.
    """
    move = moving_command_mask(obs, command_has_yaw=command_has_yaw)
    if not bool(move.any()):
        parts = [obs, actions, old_log_prob, returns, advantages]
        parts.append(old_values)
        return tuple(parts)

    src_obs = obs[move]
    src_act = actions[move]
    src_logp = old_log_prob[move]
    src_ret = returns[move]
    src_adv = advantages[move]
    mirrored_obs = mirror_observation(src_obs, command_has_yaw=command_has_yaw)
    mirrored_act = mirror_action(src_act)
    if agent is not None:
        with torch.no_grad():
            # Use device-matched mask (``action_mask`` alone stays on the construction device).
            mask = agent._mask_on(mirrored_obs) if hasattr(agent, "_mask_on") else getattr(agent, "action_mask", None)
            mirrored_logp, _ = agent.actor.log_prob(mirrored_obs, mirrored_act, mask)
            if not torch.isfinite(mirrored_logp).all():
                # Fall back to original log-probs rather than poisoning the PPO ratio.
                mirrored_logp = src_logp
    else:
        mirrored_logp = src_logp
    parts = [
        torch.cat((obs, mirrored_obs), dim=0),
        torch.cat((actions, mirrored_act), dim=0),
        torch.cat((old_log_prob, mirrored_logp), dim=0),
        torch.cat((returns, src_ret), dim=0),
        torch.cat((advantages, src_adv), dim=0),
    ]
    if old_values is not None:
        parts.append(torch.cat((old_values, old_values[move]), dim=0))
    else:
        parts.append(None)
    return tuple(parts)
