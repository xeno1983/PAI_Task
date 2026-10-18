"""Generative Adversarial Imitation Learning helpers.

Discriminator ``D(s, a)`` scores expert vs policy transitions. The policy is
rewarded with ``-log(1 - D)`` (Ho & Ermon, 2016) and updated with PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pick_and_carry.agents import mlp


class Discriminator(nn.Module):
    """Binary classifier on concatenated ``(obs, action)``."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden, 1)

    def logits(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, actions), dim=-1)).squeeze(-1)

    def probability(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(obs, actions))

    def gail_reward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``-log(1 - D(s,a))`` = softplus(logit). Higher when D thinks it is expert-like."""
        return F.softplus(self.logits(obs, actions))


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


def train_discriminator(
    disc: Discriminator,
    optimizer: torch.optim.Optimizer,
    expert_obs: torch.Tensor,
    expert_act: torch.Tensor,
    policy_obs: torch.Tensor,
    policy_act: torch.Tensor,
    *,
    epochs: int = 1,
    batch_size: int = 4096,
    action_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Binary cross-entropy: expert → 1, policy → 0."""
    expert_act = apply_action_mask(expert_act, action_mask)
    policy_act = apply_action_mask(policy_act, action_mask)
    n_e = expert_obs.shape[0]
    n_p = policy_obs.shape[0]
    total_loss = 0.0
    total_acc = 0.0
    updates = 0
    steps = max(1, (min(n_e, n_p) + batch_size - 1) // batch_size)
    for _ in range(max(1, epochs)):
        e_order = torch.randperm(n_e, device=expert_obs.device)
        p_order = torch.randperm(n_p, device=policy_obs.device)
        for step in range(steps):
            e_idx = _take_batch(e_order, step * batch_size, batch_size)
            p_idx = _take_batch(p_order, step * batch_size, batch_size)
            e_logits = disc.logits(expert_obs[e_idx], expert_act[e_idx])
            p_logits = disc.logits(policy_obs[p_idx], policy_act[p_idx])
            loss_e = F.binary_cross_entropy_with_logits(e_logits, torch.ones_like(e_logits))
            loss_p = F.binary_cross_entropy_with_logits(p_logits, torch.zeros_like(p_logits))
            loss = 0.5 * (loss_e + loss_p)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                acc = 0.5 * ((e_logits > 0).float().mean() + (p_logits < 0).float().mean())
            total_loss += float(loss.detach())
            total_acc += float(acc)
            updates += 1
    return {
        "disc_loss": total_loss / max(updates, 1),
        "disc_acc": total_acc / max(updates, 1),
    }
