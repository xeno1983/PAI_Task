"""ValueDICE-style imitation helpers (no GAN discriminator).

``NuNet`` ν(s, a) estimates a soft density-ratio dual. With next-state samples
the residual form follows ValueDICE / DualDICE:

    δ = ν(s, a) − γ ν(s′, a′)
    L = E_π[exp(δ − 1)] − E_E[ν(s, a)]

and ν is used as a dense imitation reward for PPO (Kostrikov et al., ValueDICE).
Hard clamps are avoided in the loss path; ``tanh`` soft bounds keep gradients alive.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pick_and_carry.agents import mlp

# Soft bound for ν in the dual loss / reward (gradients flow through tanh).
_NU_SCALE = 10.0
_REWARD_SCALE = 5.0


def soft_bound(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Map ℝ → (−scale, scale) with non-zero gradients everywhere."""
    return scale * torch.tanh(x / scale)


class NuNet(nn.Module):
    """Scalar ν(s, a) on concatenated observation and action."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, actions), dim=-1)).squeeze(-1)

    def nu(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Soft-bounded ν for the dual objective (gradients always flow)."""
        return soft_bound(self.forward(obs, actions), _NU_SCALE)

    def valuedice_reward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Imitation reward for PPO: softer bound than the dual ν."""
        return soft_bound(self.forward(obs, actions), _REWARD_SCALE)


def apply_action_mask(actions: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    if action_mask is None:
        return actions
    mask = action_mask.to(dtype=actions.dtype, device=actions.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    return actions * mask.expand_as(actions)


def _take_batch(order: torch.Tensor, start: int, size: int) -> torch.Tensor:
    n = order.numel()
    if n == 0:
        return order
    size = min(size, n)
    start = start % n
    end = start + size
    if end <= n:
        return order[start:end]
    return torch.cat((order[start:], order[: end - n]), dim=0)


def train_nu(
    nu_net: NuNet,
    optimizer: torch.optim.Optimizer,
    expert_obs: torch.Tensor,
    expert_act: torch.Tensor,
    policy_obs: torch.Tensor,
    policy_act: torch.Tensor,
    *,
    next_obs: torch.Tensor | None = None,
    next_act: torch.Tensor | None = None,
    gamma: float = 0.99,
    epochs: int = 1,
    batch_size: int = 4096,
    action_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Minimize the ValueDICE dual residual (with optional next-state bootstrap).

    When ``next_obs`` / ``next_act`` are provided:

        L = E_π[exp(ν(s,a) − γ ν(s′,a′) − 1)] − E_E[ν(s,a)]

    Otherwise falls back to the memoryless form ``E_π[exp(ν−1)] − E_E[ν]``.
    """
    expert_act = apply_action_mask(expert_act, action_mask)
    policy_act = apply_action_mask(policy_act, action_mask)
    use_next = next_obs is not None and next_act is not None
    if use_next:
        assert next_obs is not None and next_act is not None
        if next_obs.shape[0] != policy_obs.shape[0] or next_act.shape[0] != policy_act.shape[0]:
            raise ValueError("next_obs/next_act batch size must match policy transitions")
        next_act = apply_action_mask(next_act, action_mask)
    n_e = expert_obs.shape[0]
    n_p = policy_obs.shape[0]
    total_loss = 0.0
    total_nu_e = 0.0
    total_nu_p = 0.0
    total_delta = 0.0
    updates = 0
    steps = max(1, (min(n_e, n_p) + batch_size - 1) // batch_size)
    for _ in range(max(1, epochs)):
        e_order = torch.randperm(n_e, device=expert_obs.device)
        p_order = torch.randperm(n_p, device=policy_obs.device)
        for step in range(steps):
            e_idx = _take_batch(e_order, step * batch_size, batch_size)
            p_idx = _take_batch(p_order, step * batch_size, batch_size)
            nu_e = nu_net.nu(expert_obs[e_idx], expert_act[e_idx])
            nu_p = nu_net.nu(policy_obs[p_idx], policy_act[p_idx])
            if use_next:
                assert next_obs is not None and next_act is not None
                # Bootstrap target without flowing grads through the next ν (ValueDICE residual).
                with torch.no_grad():
                    nu_n = nu_net.nu(next_obs[p_idx], next_act[p_idx])
                delta = nu_p - float(gamma) * nu_n
            else:
                delta = nu_p
            loss = torch.exp(delta - 1.0).mean() - nu_e.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(nu_net.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_nu_e += float(nu_e.detach().mean())
            total_nu_p += float(nu_p.detach().mean())
            total_delta += float(delta.detach().mean())
            updates += 1
    denom = max(updates, 1)
    return {
        "nu_loss": total_loss / denom,
        "nu_expert": total_nu_e / denom,
        "nu_policy": total_nu_p / denom,
        "nu_delta": total_delta / denom,
    }
