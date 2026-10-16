"""Learning algorithms on top of the shared ActorCritic.

``ppo`` – clipped PPO with GAE (the default).
``grpo`` – Group Relative Policy Optimization: advantages are the return relative to
the mean of a group of parallel environments, without training the critic.
``sym_ppo`` – PPO with left–right (sagittal) data augmentation for the Go2 + Z1.
"""

from __future__ import annotations

import torch

ALGORITHMS = ("ppo", "grpo", "sym_ppo")
ALGO_LABELS = {
    "ppo": "PPO",
    "grpo": "GRPO",
    "sym_ppo": "Symmetric PPO",
}


def normalize_algo(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {"symmetric_ppo": "sym_ppo", "symmetric": "sym_ppo", "symm_ppo": "sym_ppo"}
    key = aliases.get(key, key)
    if key not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}; choose one of {ALGORITHMS}")
    return key


def group_relative_advantages(returns: torch.Tensor, group_size: int) -> torch.Tensor:
    """DeepSeek-style group relative advantages on a flat return vector.

    ``returns`` is shaped ``(N,)``. Rows are split into groups of ``group_size``
    (the last incomplete group is dropped from the relative mean and uses the
    batch mean instead). Each group is mean/std normalized.
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


def mirror_observation(obs: torch.Tensor) -> torch.Tensor:
    """Sagittal reflection of an observation batch ``(N, 81)``."""
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
    out[:, 38] *= -1.0  # arm yaw
    out[:, 44] *= -1.0  # arm yaw velocity
    out[:, 53] *= -1.0  # ee y
    out[:, 56] *= -1.0  # object y
    out[:, 59] *= -1.0  # goal y
    out[:, 67] *= -1.0  # command y
    out[:, 68] *= -1.0  # command yaw
    _swap_leg_blocks(out, 69)  # last_leg_action
    return out


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    """Sagittal reflection of an action batch ``(N, 19)``."""
    if action.shape[-1] < 19:
        raise ValueError(f"expected at least 19-dim action, got {tuple(action.shape)}")
    out = action.clone()
    _swap_leg_blocks(out, 0)
    out[:, 12] *= -1.0  # arm yaw
    return out


def augment_symmetric(
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    old_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Double a PPO batch with the left–right mirror of every sample."""
    mirrored_obs = mirror_observation(obs)
    mirrored_act = mirror_action(actions)
    parts = [
        torch.cat((obs, mirrored_obs), dim=0),
        torch.cat((actions, mirrored_act), dim=0),
        torch.cat((old_log_prob, old_log_prob), dim=0),
        torch.cat((returns, returns), dim=0),
        torch.cat((advantages, advantages), dim=0),
    ]
    if old_values is not None:
        parts.append(torch.cat((old_values, old_values), dim=0))
    else:
        parts.append(None)
    return tuple(parts)
