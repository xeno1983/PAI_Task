"""DAgger helpers for interactive imitation with ``TrotTeacher``.

Classic DAgger labels every student-visited state with the teacher action.
Gated DAgger (SafeDAgger-style) only aggregates when policy and teacher disagree
beyond a threshold, which cuts redundant labels when the policy is already good.
"""

from __future__ import annotations

import torch


def dagger_beta(round_idx: int, *, beta0: float = 1.0, decay: float = 0.85) -> float:
    """Expert-action mixing probability for env steps (not for labels).

    Round is 1-based: first round is nearly fully expert-driven.
    """
    if round_idx < 1:
        round_idx = 1
    return float(max(0.0, min(1.0, beta0 * (decay ** (round_idx - 1)))))


def action_disagreement(
    policy_act: torch.Tensor,
    teacher_act: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-env L2 disagreement on (optionally masked) action dims."""
    err = policy_act - teacher_act
    if action_mask is not None:
        mask = action_mask.to(dtype=err.dtype, device=err.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        err = err * mask.expand_as(err)
        denom = mask.sum().clamp(min=1.0).sqrt()
        return err.pow(2).sum(dim=-1).sqrt() / denom
    return err.pow(2).sum(dim=-1).sqrt()


def should_query_teacher(
    disagreement: torch.Tensor,
    *,
    threshold: float,
    mode: str = "gated",
) -> torch.Tensor:
    """Boolean mask of envs that should contribute a DAgger label.

    ``classic`` always queries; ``gated`` queries when disagreement >= threshold.
    """
    if mode == "classic":
        return torch.ones_like(disagreement, dtype=torch.bool)
    return disagreement >= float(threshold)
