"""Compact PPO and SAC agents for the vectorized pick-and-carry environment."""

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


class SquashedGaussianActor(nn.Module):
    """Gaussian policy squashed to (-1, 1) with a tanh."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.body = mlp(obs_dim, hidden)
        self.mean = nn.Linear(hidden[-1], act_dim)
        # std ≈ 0.37. Wide enough to reach down from the forward pose, narrow
        # enough that the hand does not flail into the floor on every step.
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

    def _dist(self, obs: torch.Tensor) -> Normal:
        mean = self.mean(self.body(obs))
        std = self.log_std.clamp(-5.0, 2.0).exp()
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = self._log_prob(dist, raw, action)
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action used when recording the learned policy."""
        return torch.tanh(self._dist(obs).mean)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        clipped = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(clipped)
        return self._log_prob(dist, raw, clipped), dist.entropy().sum(dim=-1)

    @staticmethod
    def _log_prob(dist: Normal, raw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        return (dist.log_prob(raw) - correction).sum(dim=-1)


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

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action used when recording the learned policy."""
        return self.mean(self.body(obs))

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        return dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1)

    def distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(self.body(obs))
        return mean, self.std().expand_as(mean)


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

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob = self.actor.sample(obs)
        value = self.value(obs)
        return action, log_prob, value

    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.mean_action(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self.actor.log_prob(obs, action)
        return log_prob, entropy, self.value(obs)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden, 1)
        self.q2 = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.actions = torch.zeros((capacity, act_dim), device=device)
        self.rewards = torch.zeros((capacity,), device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.dones = torch.zeros((capacity,), device=device)
        self.idx = 0
        self.size = 0

    def add(self, obs, actions, rewards, next_obs, dones) -> None:
        obs = obs.detach()
        actions = actions.detach()
        rewards = rewards.detach().reshape(-1)
        next_obs = next_obs.detach()
        dones = dones.detach().reshape(-1).to(dtype=torch.float32)
        n = obs.shape[0]
        if n >= self.capacity:
            obs, actions, rewards, next_obs, dones = obs[-self.capacity :], actions[-self.capacity :], rewards[-self.capacity :], next_obs[-self.capacity :], dones[-self.capacity :]
            n = self.capacity
        end = self.idx + n
        if end <= self.capacity:
            sl = slice(self.idx, end)
            self.obs[sl], self.actions[sl], self.rewards[sl], self.next_obs[sl], self.dones[sl] = obs, actions, rewards, next_obs, dones
        else:
            first = self.capacity - self.idx
            self.obs[self.idx :] = obs[:first]
            self.actions[self.idx :] = actions[:first]
            self.rewards[self.idx :] = rewards[:first]
            self.next_obs[self.idx :] = next_obs[:first]
            self.dones[self.idx :] = dones[:first]
            rest = n - first
            self.obs[:rest] = obs[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.dones[:rest] = dones[first:]
        self.idx = (self.idx + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int):
        index = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[index], self.actions[index], self.rewards[index], self.next_obs[index], self.dones[index]


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
) -> torch.Tensor:
    """KL(old || new) of diagonal Gaussians, summed over actions and averaged over the batch."""
    kl = torch.log(new_std / old_std) + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square()) - 0.5
    return kl.sum(dim=-1).mean()


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
) -> dict[str, float]:
    """One PPO update.

    ``old_values`` turns on the clipped value loss. ``desired_kl`` turns on the
    adaptive learning rate, which changes the optimizer's rate between minibatches.
    """
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
                kl = float(gaussian_kl(old_mean[batch], old_std[batch], new_mean, new_std))
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


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device, lr: float = 3e-4, alpha: float = 0.2, tau: float = 0.005):
        self.device = device
        self.alpha = alpha
        self.tau = tau
        self.actor = SquashedGaussianActor(obs_dim, act_dim).to(device)
        self.critic = TwinQ(obs_dim, act_dim).to(device)
        self.critic_target = TwinQ(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        action, _ = self.actor.sample(obs)
        return action

    def update(self, batch, gamma: float) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_target = torch.minimum(q1_t, q2_t) - self.alpha * next_log_prob
            backup = rewards + (1.0 - dones) * gamma * q_target
        q1, q2 = self.critic(obs, actions)
        critic_loss = torch.nn.functional.mse_loss(q1, backup) + torch.nn.functional.mse_loss(q2, backup)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        new_action, log_prob = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        actor_loss = (self.alpha * log_prob - torch.minimum(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
                target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": self.alpha,
        }
