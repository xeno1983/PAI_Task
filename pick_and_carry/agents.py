"""Compact PPO agent for the vectorized pick-and-carry environment."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden:
        layers.append(nn.Linear(last, width))
        layers.append(nn.ELU())
        last = width
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


MAX_LOG_STD = 0.0
MIN_LOG_STD = -5.0


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy with no squashing, as in rsl_rl and legged_gym.

    The standard deviation is a free parameter per action. It starts at ``init_std``
    and is capped at 1, so the entropy bonus cannot widen it without limit. The
    action is the raw sample. The environment scales it and clips it.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (512, 256, 128),
        init_std: float | torch.Tensor = 1.0,
    ):
        super().__init__()
        self.body = mlp(obs_dim, hidden)
        self.mean = nn.Linear(hidden[-1], act_dim)
        # Start near zero so the first actions hold the reference pose.
        nn.init.uniform_(self.mean.weight, -0.01, 0.01)
        nn.init.zeros_(self.mean.bias)
        std = torch.as_tensor(init_std, dtype=torch.float32)
        if std.ndim == 0:
            std = std.expand(act_dim).clone()
        self.log_std = nn.Parameter(std.clamp(min=1e-3).log())

    def std(self) -> torch.Tensor:
        return self.log_std.clamp(MIN_LOG_STD, MAX_LOG_STD).exp()

    def _dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mean(self.body(obs)), self.std())

    def sample(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        action = dist.sample()
        if action_mask is not None:
            # Unused dims (e.g. arm during walk) stay at 0 and do not enter the log-prob.
            action = torch.where(action_mask.expand_as(action), action, torch.zeros_like(action))
        return action, self._masked_sum(dist.log_prob(action), action_mask)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action used when recording the learned policy."""
        return self.mean(self.body(obs))

    def log_prob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        return (
            self._masked_sum(dist.log_prob(action), action_mask),
            self._masked_sum(dist.entropy(), action_mask),
        )

    def distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(self.body(obs))
        return mean, self.std().expand_as(mean)

    @staticmethod
    def _masked_sum(per_dim: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return per_dim.sum(dim=-1)
        mask = action_mask.to(dtype=per_dim.dtype, device=per_dim.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return (per_dim * mask.expand_as(per_dim)).sum(dim=-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (512, 256, 128),
        init_std: float | torch.Tensor = 1.0,
    ):
        super().__init__()
        self.actor = GaussianActor(obs_dim, act_dim, hidden, init_std)
        self.critic = mlp(obs_dim, hidden, 1)
        # Optional per-dim mask: False dims are ignored in log-prob / entropy / KL.
        self.action_mask: torch.Tensor | None = None

    def set_action_mask(self, mask: torch.Tensor | None) -> None:
        self.action_mask = None if mask is None else mask.detach().clone().bool()

    def _mask_on(self, ref: torch.Tensor) -> torch.Tensor | None:
        """Mask on ``ref``'s device (plain attribute is not moved by ``module.to``)."""
        if self.action_mask is None:
            return None
        return self.action_mask.to(device=ref.device)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob = self.actor.sample(obs, self._mask_on(obs))
        value = self.value(obs)
        return action, log_prob, value

    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.actor.mean_action(obs)
        mask = self._mask_on(action)
        if mask is not None:
            action = torch.where(mask.expand_as(action), action, torch.zeros_like(action))
        return action

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self.actor.log_prob(obs, action, self._mask_on(obs))
        return log_prob, entropy, self.value(obs)


def compute_gae(rewards, values, bootstraps, dones, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE-Lambda. ``bootstraps`` is V(s') already masked on episode ends."""
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last = torch.zeros(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * bootstraps[t] - values[t]
        last = delta + gamma * lam * mask * last
        advantages[t] = last
    return advantages, advantages + values


MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-2


def gaussian_kl(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(old || new) of diagonal Gaussians, summed over (masked) actions and averaged over the batch."""
    kl = torch.log(new_std / old_std) + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square()) - 0.5
    return GaussianActor._masked_sum(kl, action_mask).mean()


def adapt_learning_rate(learning_rate: float, kl: float, desired_kl: float) -> float:
    """rsl_rl schedule: slow down when the policy moved too far, speed up when it barely moved."""
    if kl > desired_kl * 2.0:
        learning_rate = max(MIN_LEARNING_RATE, learning_rate / 1.5)
    elif 0.0 < kl < desired_kl / 2.0:
        learning_rate = min(MAX_LEARNING_RATE, learning_rate * 1.5)
    return learning_rate


def ppo_update(
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip: float,
    epochs: int,
    minibatch_size: int,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    old_values: torch.Tensor | None = None,
    desired_kl: float | None = None,
    normalize_advantages: bool = True,
) -> dict[str, float]:
    """One PPO update.

    ``old_values`` turns on the clipped value loss. ``desired_kl`` turns on the
    adaptive learning rate, which changes the optimizer's rate between minibatches.
    Set ``normalize_advantages`` False when advantages are already group-normalized (GRPO).
    """
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = obs.shape[0]
    with torch.no_grad():
        old_mean, old_std = agent.actor.distribution_params(obs)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(n, device=obs.device)
        for start in range(0, n, minibatch_size):
            batch = order[start : start + minibatch_size]
            log_prob, entropy, value = agent.evaluate(obs[batch], actions[batch])
            ratio = torch.exp(log_prob - old_log_prob[batch])
            adv = advantages[batch]
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv).mean()
            if old_values is None:
                value_loss = 0.5 * (returns[batch] - value).pow(2).mean()
            else:
                old = old_values[batch]
                clipped = old + (value - old).clamp(-clip, clip)
                value_loss = 0.5 * torch.maximum((returns[batch] - value).pow(2), (returns[batch] - clipped).pow(2)).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                new_mean, new_std = agent.actor.distribution_params(obs[batch])
                kl = float(
                    gaussian_kl(
                        old_mean[batch],
                        old_std[batch],
                        new_mean,
                        new_std,
                        action_mask=agent.action_mask,
                    )
                )
            if desired_kl is not None:
                for group in optimizer.param_groups:
                    group["lr"] = adapt_learning_rate(float(group["lr"]), kl, desired_kl)
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy_mean.detach())
            totals["kl"] += kl
            updates += 1
    metrics = {key: value / max(updates, 1) for key, value in totals.items()}
    metrics["lr"] = float(optimizer.param_groups[0]["lr"])
    return metrics

