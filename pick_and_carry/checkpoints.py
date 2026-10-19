"""Checkpoint files: saving, finding the newest compatible one, and moving stale ones aside.

A checkpoint records the network shape (observation and action size), the arm, and the
curriculum stage. A folder's own weights win; an empty folder borrows from an easier
stage, and ``retreat`` borrows from a harder one.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from pick_and_carry.config import layout_dim
from pick_and_carry.curriculum import (
    CATCH_LEVELS,
    earlier_stages,
    later_stages,
    stage_layouts,
    stage_log_dir,
)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_iteration(path: Path) -> int:
    """Numeric iteration from ``ppo_1234.pt``. Non-matching names sort as -1."""
    stem = path.stem
    if stem.startswith("ppo_") and stem[4:].isdigit():
        return int(stem[4:])
    return -1


_IMITATION_FLAGS = ("bc_trot", "dagger_trot", "valuedice_trot", "gail_trot")


def is_imitation_checkpoint(blob: dict) -> bool:
    """True for BC / DAgger / ValueDICE / GAIL warm-start files."""
    return any(bool(blob.get(flag)) for flag in _IMITATION_FLAGS)


def is_rl_training_checkpoint(blob: dict) -> bool:
    """True for on-policy RL snapshots (PPO / GRPO / Sym) that carry training progress.

    Distinguishes them from imitation warm-starts so ``--init-from`` can continue
    iteration instead of silently resetting to 0.
    """
    if is_imitation_checkpoint(blob) or "model" not in blob:
        return False
    if "optimizer" in blob:
        return True
    if "iteration" not in blob:
        return False
    algo = str(blob.get("algo", "") or "").lower().replace("-", "_")
    return algo in ("ppo", "grpo", "sym_ppo", "symmetric_ppo", "")


def iter_sorted_checkpoints(log_dir: Path) -> list[Path]:
    """Checkpoints ordered by iteration number (not lexicographic name).

    Lex sort puts ``ppo_9980.pt`` after ``ppo_29980.pt``, so ``--resume latest``
    would reload the wrong file and overwrite later weights.
    """
    if not log_dir.is_dir():
        return []
    return sorted(log_dir.glob("ppo_*.pt"), key=checkpoint_iteration)


def checkpoint_matches(
    blob: dict,
    obs_dim: int,
    act_dim: int,
    arm: str,
    curriculum: str,
    *,
    match_arm: bool,
) -> bool:
    if int(blob.get("obs_dim", -1)) != obs_dim or int(blob.get("act_dim", -1)) != act_dim:
        return False
    # Files without an arm field (old layouts) do not load.
    if match_arm and ("arm" not in blob or str(blob["arm"]) != arm):
        return False
    # Files saved before the curriculum field are pick-and-place.
    return str(blob.get("curriculum", "carry")) == curriculum


def latest_matching(
    log_dir: Path,
    obs_dim: int,
    act_dim: int,
    arm: str,
    curriculum: str,
    *,
    match_arm: bool = True,
) -> Path | None:
    """Newest checkpoint in one folder whose network matches this task."""
    if not log_dir.is_dir():
        return None
    compatible: list[Path] = []
    skipped_arm = 0
    for path in iter_sorted_checkpoints(log_dir):
        blob = torch_load(path, torch.device("cpu"))
        saved_arm = str(blob["arm"]) if "arm" in blob else ""
        if match_arm and saved_arm != arm and int(blob.get("obs_dim", -1)) == obs_dim:
            skipped_arm += 1
        if checkpoint_matches(blob, obs_dim, act_dim, arm, curriculum, match_arm=match_arm):
            compatible.append(path)
    if not compatible and skipped_arm:
        print(f"skipped {skipped_arm} checkpoint(s) from a different arm.", flush=True)
    return compatible[-1] if compatible else None


def level_from_path(path: Path) -> str:
    name = path.parent.name
    for algo in ("ppo", "grpo", "sym_ppo"):
        prefix = f"{algo}_curriculum_catch_"
        if name.startswith(prefix) and name[len(prefix) :] in CATCH_LEVELS:
            return name[len(prefix) :]
    return "static"


def stage_rank(stage: str, catch_level: str) -> int:
    """Higher means a harder task on the same ladder. Carry stays at zero."""
    ladder = ("walk", "reach", "haul")
    if stage in ladder:
        return 1 + ladder.index(stage)
    if stage == "catch" and catch_level in CATCH_LEVELS:
        return 1 + len(ladder) + CATCH_LEVELS.index(catch_level)
    return 0


def resolve_checkpoint(
    log_dir: Path,
    resume: str,
    obs_dim: int,
    act_dim: int,
    arm: str,
    curriculum: str = "carry",
    catch_level: str = "static",
    root: Path | None = None,
    retreat: bool = False,
    algo: str = "ppo",
) -> Path | None:
    """Own folder first. Empty folders use an easier stage. ``retreat`` prefers a harder stage."""
    if resume != "latest":
        path = Path(resume)
        if not path.is_file():
            raise SystemExit(f"Checkpoint not found: {path}")
        return path
    match_arm = True
    if retreat:
        for stage, level in later_stages(curriculum, catch_level):
            found = latest_matching(
                stage_log_dir(stage, level, arm, root=root, algo=algo),
                obs_dim,
                act_dim,
                arm,
                stage,
                match_arm=match_arm,
            )
            if found is not None:
                return found
    own = latest_matching(log_dir, obs_dim, act_dim, arm, curriculum)
    if own is not None:
        return own
    for stage, level in earlier_stages(curriculum, catch_level):
        found = latest_matching(
            stage_log_dir(stage, level, arm, root=root, algo=algo),
            obs_dim,
            act_dim,
            arm,
            stage,
            match_arm=match_arm,
        )
        if found is not None:
            return found
    return None


def missing_play_checkpoint(args, cfg, root: Path | None = None) -> str | None:
    """Japanese reason to stop before building a scene, when playback has no weights."""
    if not getattr(args, "play", False) or not getattr(args, "resume", None):
        return None
    from pick_and_carry.arms import get_arm

    arm = get_arm(cfg.arm)
    observation, action = stage_layouts(cfg.curriculum, arm.dof)
    algo = getattr(args, "algo", "ppo")
    found = resolve_checkpoint(
        args.log_dir,
        args.resume,
        layout_dim(observation),
        layout_dim(action),
        cfg.arm,
        cfg.curriculum,
        cfg.catch_level,
        root=root,
        retreat=getattr(args, "retreat", False),
        algo=algo,
    )
    if found is not None:
        return None
    looked = [str(Path(args.log_dir))]
    for stage, level in earlier_stages(cfg.curriculum, cfg.catch_level):
        looked.append(str(stage_log_dir(stage, level, cfg.arm, root=root, algo=algo)))
    lines = [
        "再生できる重みがありません。",
        "探した場所: " + "、".join(looked),
        "先にメニューの 1 で、この段階を学習してください。",
    ]
    return "\n".join(lines)


def save_checkpoint(
    path: Path,
    agent,
    optimizer,
    iteration: int,
    obs_dim: int,
    act_dim: int,
    arm: str,
    curriculum: str = "carry",
    algo: str = "ppo",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "arm": arm,
            "curriculum": curriculum,
            "algo": algo,
        },
        path,
    )
    print(f"saved {path} (iteration {iteration})", flush=True)


def archive_incompatible_logs(log_dir: Path, obs_dim: int, act_dim: int, arm: str, curriculum: str) -> Path | None:
    """Move weights and curves of another network shape aside so a new run cannot overwrite them.

    Nothing moves when a compatible checkpoint exists or the folder holds no checkpoint.
    """
    weights = iter_sorted_checkpoints(log_dir)
    if not weights:
        return None
    if latest_matching(log_dir, obs_dim, act_dim, arm, curriculum) is not None:
        return None
    target = log_dir / f"old_{time.strftime('%Y%m%d_%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    moving = [*weights, *log_dir.glob("events.out.tfevents*")]
    metrics = log_dir / "metrics.jsonl"
    if metrics.exists():
        moving.append(metrics)
    for path in moving:
        path.replace(target / path.name)
    return target
