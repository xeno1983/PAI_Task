"""On-policy PPO loop for the Genesis pick-and-carry environment.

The environment auto-resets, so a timed-out transition bootstraps with V(s_t)
added into the reward and is then treated as terminal. Falls and successful
placements bootstrap with zero.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pick_and_carry.agents import ActorCritic, GaussianActor, compute_gae, ppo_update
from pick_and_carry.algorithms import (
    ALGORITHMS,
    ALGO_LABELS,
    augment_symmetric,
    monte_carlo_returns,
    normalize_algo,
    stabilize_grpo_advantages,
)
from pick_and_carry.checkpoints import (
    archive_incompatible_logs,
    is_imitation_checkpoint,
    is_rl_training_checkpoint,
    level_from_path,
    missing_play_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
    stage_rank,
    torch_load,
)
from pick_and_carry.config import EnvConfig
from pick_and_carry.difficulty import Difficulty
from pick_and_carry.curriculum import (
    CATCH_HOLD_SECONDS,
    CATCH_LEVELS,
    LOCOMOTION_STAGES,
    STAGES,
    holds_arm_pose,
    locomotion_action_mask,
    stage_log_dir,
)
from pick_and_carry.console import format_metric
from pick_and_carry.env import PickAndCarryEnv
from pick_and_carry.ros_bridge import FileBridge, pack_rgb_image, ros_usage_error
from pick_and_carry.mlflow_log import LocalTracker
from pick_and_carry.tracking import append_metrics, write_run


def format_gpu_memory(free_bytes: int, total_bytes: int) -> str:
    """Used and total GPU memory. Task Manager's process memory does not show this."""
    used = max(0, total_bytes - free_bytes)
    return f"GPU {used / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GB"


def format_remaining(seconds: float) -> str:
    """Predicted time left, as hours, minutes, and seconds."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}時間{minutes:02d}分{secs:02d}秒"
    return f"{minutes}分{secs:02d}秒"


def minibatch_size_for(args, batch: int) -> int:
    """Samples per minibatch: the explicit size, else the batch split into --num-mini-batches."""
    if args.minibatch_size > 0:
        return int(args.minibatch_size)
    return max(1, batch // max(1, int(args.num_mini_batches)))


def initial_action_std(num_actions: int, leg_std: float, arm_std: float, cpg_count: int | None = None):
    """Exploration noise per action: wider on CPG params, narrower on arm/gripper."""
    from pick_and_carry.config import CPG_PARAM_DIM

    count = CPG_PARAM_DIM if cpg_count is None else int(cpg_count)
    std = torch.full((num_actions,), float(arm_std))
    std[:count] = float(leg_std)
    return std


def parse_args():
    parser = argparse.ArgumentParser(description="Train pick-and-carry on Genesis.")
    parser.add_argument(
        "--algo",
        choices=ALGORITHMS,
        default="ppo",
        help="ppo: clipped PPO. grpo: group-relative advantages without critic. "
        "sym_ppo: PPO with left-right observation/action augmentation.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="GRPO group size. 0 uses --num-envs.",
    )
    parser.add_argument(
        "--grpo-adv-clip",
        type=float,
        default=5.0,
        help="GRPO: clamp group-relative advantages to ±this (0 disables).",
    )
    parser.add_argument(
        "--grpo-return-quantile",
        type=float,
        default=0.02,
        help="GRPO: winsorize MC returns at [q, 1-q] before grouping (0 disables).",
    )
    parser.add_argument(
        "--grpo-entropy-coef",
        type=float,
        default=0.002,
        help="GRPO entropy bonus (default 0.002; PPO still uses --entropy-coef).",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--num-steps", type=int, default=128)
    # Defaults follow rsl_rl / legged_gym: lr starts at 1e-3 and follows the KL target.
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--desired-kl",
        type=float,
        default=0.015,
        help="Adaptive learning rate target. 0 keeps --lr fixed.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=0, help="Samples per minibatch. 0 uses batch / --num-mini-batches.")
    # Higher than classic 0.001 so a BC warm-start can keep exploring instead of freezing.
    parser.add_argument("--entropy-coef", type=float, default=0.008)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--leg-init-std", type=float, default=0.4)
    parser.add_argument("--arm-init-std", type=float, default=0.4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record", type=str, default=None, help="Save an mp4 of env 0 to this path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain-amplitude", type=float, default=None, help="Bump and step height in meters.")
    parser.add_argument("--terrain-steps", action="store_true", help="Add flat treads on the travel strips.")
    parser.add_argument(
        "--arm",
        choices=("z1",),
        default="z1",
        help="Unitree Z1. Observation 81, action CPG+arm (see config.NUM_ACTIONS).",
    )
    parser.add_argument(
        "--control-split",
        choices=("split", "walk", "both"),
        default="split",
        help="split: legs then arm. walk: locomotion only. both: legs and arm together.",
    )
    parser.add_argument(
        "--wrist-camera",
        action="store_true",
        help="Detect the cube with a camera on the hand and grasp only when it is seen.",
    )
    parser.add_argument(
        "--curriculum",
        choices=STAGES,
        default="carry",
        help="walk (stand, walk, turn, bumpy ground, arm moving), then reach, then haul and carry (the box) "
        "or catch.",
    )
    parser.add_argument(
        "--easy-walk",
        action="store_true",
        help="Walk only: flat ground, no pushes / friction noise. Saves under *_walk_easy.",
    )
    parser.add_argument(
        "--arm-disturb",
        action="store_true",
        help="Walk only: randomly wander the held arm (off by default; opt-in robustness).",
    )
    parser.add_argument(
        "--no-arm-disturb",
        action="store_true",
        help="Walk only: keep the arm at the reset pose (default; kept for older scripts).",
    )
    parser.add_argument(
        "--catch-level",
        choices=CATCH_LEVELS,
        default="static",
        help="How the ball moves in the catch stage: static, rolling, thrown, thrown with a noisy view.",
    )
    parser.add_argument(
        "--retreat",
        action="store_true",
        help="Practice this easier stage from a harder stage's weights.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/ppo"))
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Checkpoint path, or 'latest' in --log-dir. With no value, 'latest' is used.",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Warm-start actor/critic. Imitation (bc/dagger/…) starts at iteration 0. "
        "An RL ppo_*.pt in the same --log-dir continues that iteration (resume semantics). "
        "Aliases: bc / dagger / valuedice / gail / latest (logs/imitation + walk folders). "
        "Ignored when --resume finds a checkpoint in --log-dir.",
    )
    parser.add_argument(
        "--no-auto-init-from",
        action="store_true",
        help="Do not auto-load dagger/bc/valuedice/gail weights when --easy-walk and no --init-from.",
    )
    parser.add_argument(
        "--bc-imit-coef",
        type=float,
        default=None,
        help="Add −coef·MSE(a, a_BC(s)) to the env reward (masked dims ignored). "
        "Default: 0.1 after --init-from / easy-walk auto warm-start, else 0. Pass 0 to disable.",
    )
    parser.add_argument("--play", action="store_true", help="Run the loaded policy and skip learning.")
    parser.add_argument("--seconds", type=float, default=20.0, help="Length of a --play recording, in simulation seconds.")
    parser.add_argument("--object-xy", type=str, default=None, help="Inference supply position, 'x,y' in meters.")
    parser.add_argument("--goal-xy", type=str, default=None, help="Inference drop position, 'x,y' in meters.")
    parser.add_argument(
        "--ros",
        action="store_true",
        help="Inference only. Publish robot state and accept a new drop point while playing.",
    )
    parser.add_argument(
        "--ros-dir",
        type=Path,
        default=Path("logs/ros_ipc"),
        help="Folder shared with rviz2. Poses and goal clicks are files in here.",
    )
    return parser.parse_args()


def _parse_xy(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    parts = text.split(",")
    if len(parts) != 2:
        raise SystemExit(f"Expected x,y, got {text}")
    return float(parts[0]), float(parts[1])


def _latest_checkpoint_iteration(log_dir: Path, resume: str | None) -> int:
    """Best-effort iteration for naming play videos before the env exists."""
    import re

    if resume and resume != "latest":
        path = Path(resume)
        match = re.search(r"(\d{4,})", path.stem)
        if match:
            return int(match.group(1))
        return 0
    if not log_dir.is_dir():
        return 0
    best = 0
    for path in log_dir.glob("ppo_*.pt"):
        match = re.search(r"(\d+)", path.stem)
        if match:
            best = max(best, int(match.group(1)))
    for name in ("dagger_trot.pt", "bc_trot.pt", "valuedice_trot.pt", "gail_trot.pt"):
        if (log_dir / name).is_file() and best == 0 and resume in (None, "latest", name, str(log_dir / name)):
            return 0
    return best


def prepare_record_path(args) -> None:
    """Nest ``--record`` under ``log_dir/videos/[imitation/]{task}/{kind}_iter…_stamp/``."""
    if not args.record:
        return
    from pick_and_carry.video_paths import is_under_videos_tree, make_video_run_dir, video_task_name

    log_dir = Path(args.log_dir).resolve()
    args.log_dir = log_dir
    requested = Path(args.record)
    # Already nested under this log_dir's videos/ tree — keep as-is (ensure parent exists).
    if is_under_videos_tree(requested, log_dir):
        requested.parent.mkdir(parents=True, exist_ok=True)
        args.record = str(requested.resolve())
        return
    easy = bool(getattr(args, "easy_walk", False) or "walk_easy" in log_dir.name)
    task = video_task_name(
        getattr(args, "curriculum", "carry"),
        easy=easy,
        catch_level=getattr(args, "catch_level", "static"),
    )
    iteration = _latest_checkpoint_iteration(log_dir, getattr(args, "resume", None))
    kind = normalize_algo(getattr(args, "algo", "ppo"))
    resume = str(getattr(args, "resume", "") or "")
    if "bc_trot" in resume:
        kind = "bc"
    elif "dagger_trot" in resume:
        kind = "dagger"
    elif "valuedice_trot" in resume:
        kind = "valuedice"
    elif "gail_trot" in resume:
        kind = "gail"
    run_dir = make_video_run_dir(
        log_dir,
        task=task,
        kind=kind,
        iteration=iteration,
        extra={"requested_name": requested.name, "resume": resume},
    )
    nested = (run_dir / requested.name).resolve()
    args.record = str(nested)
    print(f"record → {nested}", flush=True)


def make_config(args) -> EnvConfig:
    amplitude = 0.035 if args.terrain_amplitude is None else args.terrain_amplitude
    cfg = EnvConfig(
        num_envs=args.num_envs,
        backend=args.backend,
        show_viewer=args.viewer,
        record_path=args.record,
        seed=args.seed,
        terrain_steps=args.terrain_steps,
        terrain_amplitude=amplitude,
        control_split=args.control_split,
        arm=args.arm,
        wrist_camera=args.wrist_camera,
        fixed_object_xy=_parse_xy(args.object_xy),
        fixed_goal_xy=_parse_xy(args.goal_xy),
        curriculum=getattr(args, "curriculum", "carry"),
        catch_level=getattr(args, "catch_level", "static"),
    )
    if holds_arm_pose(cfg.curriculum):
        cfg.control_split = "walk"
    elif cfg.curriculum != "carry":
        cfg.control_split = "both"
    easy_walk = bool(getattr(args, "easy_walk", False)) and cfg.curriculum in LOCOMOTION_STAGES
    if cfg.curriculum in LOCOMOTION_STAGES:
        # Walk is the one stage on bumpy ground. The bumps start flat and grow with the score.
        cfg.terrain = not easy_walk
        # Arm stays at reset pose unless --arm-disturb (held pose; policy does not drive it).
        cfg.arm_disturb = bool(getattr(args, "arm_disturb", False)) and not bool(
            getattr(args, "no_arm_disturb", False)
        )
        if easy_walk:
            cfg.terrain_steps = False
            cfg.terrain_amplitude = 0.0
    elif cfg.curriculum != "carry":
        cfg.terrain = False
        cfg.terrain_steps = False
    if cfg.curriculum in ("haul", "reach", "catch"):
        cfg.catch_hold_steps = max(1, int(round(CATCH_HOLD_SECONDS / cfg.dt)))
    if cfg.curriculum in LOCOMOTION_STAGES and not args.play and not easy_walk:
        # Pushes and per-robot friction, as in legged_gym. Playback stays undisturbed.
        cfg.push_robots = True
        cfg.randomize_friction = True
    if args.play and args.ros:
        # One click session should not snap the robot home on the 20 s training timeout.
        cfg.episode_length_s = max(cfg.episode_length_s, float(args.seconds))
        cfg.enable_camera = True
    if not args.play and args.num_envs > 1 and not cfg.wrist_camera:
        cfg.fleet_snapshot = str(Path(args.log_dir) / "fleet.png")
    return cfg


def resume_agent(args, env, cfg, agent, optimizer) -> tuple[int, str | None]:
    """Load a checkpoint when --resume is set. Returns the saved iteration and path."""
    if not args.resume:
        return 0, None
    ckpt_path = resolve_checkpoint(
        args.log_dir,
        args.resume,
        env.num_obs,
        env.num_actions,
        cfg.arm,
        cfg.curriculum,
        cfg.catch_level,
        retreat=getattr(args, "retreat", False),
        algo=getattr(args, "algo", "ppo"),
    )
    if ckpt_path is None:
        if args.play:
            raise SystemExit(
                "再生できる重みがありません。先にメニューの 1 で、この段階を学習してください。"
            )
        print(f"{args.log_dir.resolve()} にも前の段階にも重みがありません。最初から学習します。", flush=True)
        return 0, None
    blob = torch_load(ckpt_path, env.device)
    obs_dim = int(blob.get("obs_dim", -1))
    act_dim = int(blob.get("act_dim", -1))
    saved_arm = str(blob["arm"]) if "arm" in blob else ""
    saved_stage = str(blob.get("curriculum", "carry"))
    same_network = (
        obs_dim == env.num_obs
        and act_dim == env.num_actions
        and saved_arm == cfg.arm
    )
    if not same_network:
        print(
            f"checkpoint {ckpt_path} is for arm {saved_arm or '?'} obs {obs_dim} act {act_dim}, "
            f"this task is arm {cfg.arm} obs {env.num_obs} act {env.num_actions}. Starting from scratch.",
            flush=True,
        )
        return 0, None
    agent.load_state_dict(blob["model"])
    same_folder = ckpt_path.parent.resolve() == Path(args.log_dir).resolve()
    if not same_folder:
        harder = stage_rank(saved_stage, level_from_path(ckpt_path)) > stage_rank(cfg.curriculum, cfg.catch_level)
        if harder:
            print(f"上の段階の重みで、易しい課題に戻って学びます: {ckpt_path}", flush=True)
        else:
            print(f"前の段階 {saved_stage} の重みから始めます: {ckpt_path}", flush=True)
        return 0, str(ckpt_path)
    if "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    start_iteration = int(blob.get("iteration", 0))
    print(f"resumed {ckpt_path} at iteration {start_iteration}", flush=True)
    return start_iteration, str(ckpt_path)


def resolve_init_from(init_from: str | Path | None, log_dir: Path) -> Path | None:
    """Resolve ``--init-from`` path or alias (``bc``, ``dagger``, ``latest``, …).

    Aliases search ``logs/imitation/`` first, then legacy nested/flat walk folders.
    """
    from pick_and_carry.curriculum import imitation_weight_dirs

    if init_from is None:
        return None
    text = str(init_from).strip()
    if not text:
        return None
    aliases = {
        "bc": ("bc_trot.pt",),
        "bc_trot": ("bc_trot.pt",),
        "dagger": ("dagger_trot.pt",),
        "dagger_trot": ("dagger_trot.pt",),
        "valuedice": ("valuedice_trot.pt",),
        "valuedice_trot": ("valuedice_trot.pt",),
        "gail": ("gail_trot.pt", "valuedice_trot.pt"),
        "gail_trot": ("gail_trot.pt", "valuedice_trot.pt"),
        "latest": ("dagger_trot.pt", "bc_trot.pt", "valuedice_trot.pt", "gail_trot.pt"),
    }
    key = text.lower().replace("-", "_")
    names = aliases.get(key)
    search = imitation_weight_dirs(log_dir)
    if names is not None:
        for folder in search:
            for name in names:
                path = folder / name
                if path.is_file():
                    return path
        searched = ", ".join(str(d) for d in search)
        raise SystemExit(f"--init-from {text} が見つかりません（探した場所: {searched}）")
    path = Path(text)
    if not path.is_file():
        for folder in search:
            alt = folder / text
            if alt.is_file():
                return alt
        raise SystemExit(f"--init-from not found: {path}")
    return path


def _checkpoint_matches_env(blob: dict, env, cfg) -> None:
    obs_dim = int(blob.get("obs_dim", -1))
    act_dim = int(blob.get("act_dim", -1))
    saved_arm = str(blob["arm"]) if "arm" in blob else ""
    if obs_dim != env.num_obs or act_dim != env.num_actions or saved_arm != cfg.arm:
        raise SystemExit(
            f"checkpoint is arm {saved_arm or '?'} obs {obs_dim} act {act_dim}, "
            f"need arm {cfg.arm} obs {env.num_obs} act {env.num_actions}."
        )


def load_rl_checkpoint_via_init_from(
    path: Path,
    env,
    cfg,
    agent,
    optimizer,
    log_dir: Path,
) -> tuple[int, str]:
    """Load a PPO/GRPO checkpoint passed through ``--init-from``.

    Same log folder → continue iteration and optimizer (resume semantics).
    Other folder → weights only, iteration 0 (stage transfer), no imitation flag.
    """
    ckpt = Path(path)
    blob = torch_load(ckpt, env.device)
    _checkpoint_matches_env(blob, env, cfg)
    agent.load_state_dict(blob["model"])
    same_folder = ckpt.parent.resolve() == Path(log_dir).resolve()
    if same_folder:
        if "optimizer" in blob:
            optimizer.load_state_dict(blob["optimizer"])
        start_iteration = int(blob.get("iteration", 0))
        print(
            f"--init-from RL checkpoint treated as resume: {ckpt} at iteration {start_iteration}",
            flush=True,
        )
        return start_iteration, str(ckpt)
    print(
        f"--init-from RL checkpoint from another folder: {ckpt} "
        f"(weights only, iteration 0 — like stage transfer)",
        flush=True,
    )
    return 0, str(ckpt)


def load_init_weights(
    path: Path,
    env,
    cfg,
    agent,
    *,
    reset_log_std: torch.Tensor | float | None = None,
) -> str | None:
    """Load actor/critic weights without optimizer or iteration (BC warm-start).

    ``reset_log_std`` refreshes the exploration noise after load (keeps BC means).
    Imitation checkpoints often carry a tight ``log_std`` that otherwise freezes PPO.
    Do not use this for RL ``ppo_*.pt`` files — see ``load_rl_checkpoint_via_init_from``.
    """
    if path is None:
        return None
    ckpt = Path(path)
    if not ckpt.is_file():
        raise SystemExit(f"--init-from not found: {ckpt}")
    blob = torch_load(ckpt, env.device)
    _checkpoint_matches_env(blob, env, cfg)
    if is_rl_training_checkpoint(blob):
        raise SystemExit(
            f"--init-from {ckpt} looks like an RL training checkpoint; "
            "internal routing should have used resume semantics."
        )
    agent.load_state_dict(blob["model"])
    if reset_log_std is not None:
        with torch.no_grad():
            std = torch.as_tensor(reset_log_std, dtype=torch.float32, device=agent.actor.log_std.device)
            if std.ndim == 0:
                std = std.expand_as(agent.actor.log_std)
            agent.actor.log_std.copy_(std.clamp(min=1e-3).log())
        print(
            f"reset actor log_std after warm-start (std mean={float(std.mean()):.3f})",
            flush=True,
        )
    if blob.get("dagger_trot"):
        kind = "DAgger"
    elif blob.get("valuedice_trot"):
        kind = "ValueDICE"
    elif blob.get("gail_trot"):
        kind = "GAIL"
    elif blob.get("bc_trot"):
        kind = "BC trot"
    else:
        kind = "init"
    print(f"{kind} weights from {ckpt} (PPO starts at iteration 0)", flush=True)
    return str(ckpt)


def freeze_actor_as_bc_teacher(actor: GaussianActor) -> GaussianActor:
    """Frozen copy of the warm-start actor; only ``mean_action`` is used for imitation reward."""
    teacher = copy.deepcopy(actor)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def bc_imitation_reward(
    teacher: GaussianActor,
    obs: torch.Tensor,
    action: torch.Tensor,
    action_mask: torch.Tensor | None,
    coef: float,
) -> torch.Tensor:
    """−coef · mean squared error vs frozen BC mean (masked dims excluded from the mean)."""
    if coef <= 0.0:
        return torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
    with torch.no_grad():
        target = teacher.mean_action(obs)
    diff = action - target
    if action_mask is not None:
        mask = action_mask.to(dtype=diff.dtype, device=diff.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(diff)
        diff = diff * mask
        denom = mask.sum(dim=-1).clamp(min=1.0)
        mse = diff.square().sum(dim=-1) / denom
    else:
        mse = diff.square().mean(dim=-1)
    return (-float(coef)) * mse


def resolve_bc_imit_coef(args, *, warmed_from_imitation: bool) -> float:
    """Explicit ``--bc-imit-coef`` wins; else 0.1 after imitation warm-start, else 0."""
    explicit = getattr(args, "bc_imit_coef", None)
    if explicit is not None:
        return float(explicit)
    return 0.1 if warmed_from_imitation else 0.0


_live_window_ready = False


def _show_live(frame, goal_xy: tuple[float, float]) -> None:
    """Open a normal window with the overview. The Genesis viewer stays closed."""
    global _live_window_ready
    try:
        import cv2

        image = np.ascontiguousarray(frame[..., ::-1])
        cv2.putText(
            image,
            f"goal {goal_xy[0]:.2f}, {goal_xy[1]:.2f}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 220, 80),
            2,
            cv2.LINE_AA,
        )
        if not _live_window_ready:
            cv2.namedWindow("Pick-and-Carry", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Pick-and-Carry", 960, 540)
            _live_window_ready = True
            print("シミュレーション画面を開きました。", flush=True)
        cv2.imshow("Pick-and-Carry", image)
        cv2.waitKey(1)
    except Exception as exc:
        if not _live_window_ready:
            print(f"シミュレーション画面を開けません。{exc}", flush=True)
            _live_window_ready = True


def _close_live() -> None:
    if not _live_window_ready:
        return
    try:
        import cv2

        cv2.destroyWindow("Pick-and-Carry")
    except Exception:
        pass


def _accept_goal(env, obs, bridge: FileBridge, step_index: int):
    """Apply a clicked drop point before the policy acts, then queue the current pose."""
    goal = bridge.poll_goal()
    if goal is not None:
        current_x = float(env.goal_pos[0, 0])
        current_y = float(env.goal_pos[0, 1])
        if abs(current_x - goal[0]) > 0.02 or abs(current_y - goal[1]) > 0.02:
            applied = env.set_goal_xy(*goal)
            obs = env.obs_buf
            print(f"運び先を ({applied[0]:.2f}, {applied[1]:.2f}) に更新しました。", flush=True)
    state = env.inference_state()
    if step_index % 2 == 0:
        frame = env.overview_rgb()
        if frame is not None:
            state["image"] = pack_rgb_image(frame, stride=1)
            _show_live(frame, (float(env.goal_pos[0, 0]), float(env.goal_pos[0, 1])))
    bridge.publish(state)
    return obs


def play_policy(args, cfg, env, agent, obs) -> None:
    if not args.resume:
        raise SystemExit("Pass --resume latest (or a checkpoint path) with --play.")
    bridge = None
    steps = max(1, int(round(args.seconds / cfg.dt)))
    print(f"playing {steps} steps ({args.seconds:.0f}s)", flush=True)
    env.record_initial_pose()
    try:
        if args.ros:
            bridge = FileBridge(args.ros_dir)
            bridge.start()
            print(
                f"rviz2 と {bridge.directory.resolve()} で姿勢と運び先を受け渡します。"
                "シミュレーション画面が開き、Publish Point で運び先を変えます。",
                flush=True,
            )
        for step_index in range(steps):
            if bridge is not None:
                obs = _accept_goal(env, obs, bridge, step_index)
            with torch.no_grad():
                action = agent.act_mean(obs)
            obs, _, _, _ = env.step(action)
        if bridge is not None:
            _accept_goal(env, obs, bridge, steps)
    finally:
        _close_live()
        if bridge is not None:
            bridge.close()


def run_training(
    args,
    cfg,
    env,
    agent,
    optimizer,
    writer,
    obs,
    start_iteration: int,
    tracker: LocalTracker,
    *,
    bc_teacher: GaussianActor | None = None,
    bc_imit_coef: float = 0.0,
) -> None:
    if start_iteration == 0:
        first = args.log_dir / "ppo_000000.pt"
        save_checkpoint(
            first,
            agent,
            optimizer,
            0,
            env.num_obs,
            env.num_actions,
            cfg.arm,
            cfg.curriculum,
            algo=normalize_algo(getattr(args, "algo", "ppo")),
        )
        tracker.log_checkpoint(first)
    last_iteration = start_iteration + args.iterations
    if bc_teacher is not None and bc_imit_coef > 0.0:
        print(f"BC imitation reward coef={bc_imit_coef:.3f} (frozen warm-start mean)", flush=True)
    print(
        f"学習 {start_iteration}/{last_iteration} から開始します。1回終わるたびに残り時間を出します。",
        flush=True,
    )
    step_times: list[float] = []
    prev_log: dict[str, float] | None = None
    difficulty = Difficulty(cfg.curriculum, cfg.catch_level)
    difficulty.start(env)
    for iteration in range(start_iteration + 1, last_iteration + 1):
        started = time.perf_counter()
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, boot_buf, done_buf = [], [], [], [], [], [], []
        grasp_sum = 0.0
        dist_sum = 0.0
        gait_sum = 0.0
        steady_sum = 0.0
        jerk_sum = 0.0
        tracking_sum = 0.0
        grip_sum = 0.0
        success_sum = 0.0
        success_count = 0
        bc_imit_sum = 0.0
        if cfg.curriculum in (*LOCOMOTION_STAGES, "haul"):
            progress_key = "walk_score"
        elif cfg.curriculum in ("catch", "reach"):
            progress_key = "approach"
        else:
            progress_key = "ee_dist"
        algo = normalize_algo(getattr(args, "algo", "ppo"))
        for _ in range(args.num_steps):
            with torch.no_grad():
                action, log_prob, value = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                if bc_teacher is not None and bc_imit_coef > 0.0:
                    imit = bc_imitation_reward(
                        bc_teacher,
                        obs,
                        action,
                        agent._mask_on(obs),
                        bc_imit_coef,
                    )
                    reward = reward + imit
                    bc_imit_sum += float(imit.mean())
                if algo == "grpo":
                    # Critic-free: no value bootstrap into the reward or the return.
                    bootstrap = torch.zeros_like(reward)
                else:
                    # Approximate V(s') with V(s) on timeout, because step() already reset.
                    reward = reward + args.gamma * value * info["time_outs"]
                    next_value = agent.value(next_obs)
                    bootstrap = next_value * (~done).to(dtype=reward.dtype)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            val_buf.append(value)
            rew_buf.append(reward)
            boot_buf.append(bootstrap)
            done_buf.append(done.to(dtype=reward.dtype))
            obs = next_obs
            grasp_sum += float(env.grasped.float().mean())
            if "success_rate" in info:
                success_sum += float(info["success_rate"])
                success_count += 1
            progress = env.reward_terms.get(progress_key)
            if progress is not None:
                dist_sum += float(progress.mean())
            gait_score = env.reward_terms.get("gait_score")
            if gait_score is not None:
                gait_sum += float(gait_score.mean())
            steady_score = env.reward_terms.get("steady")
            if steady_score is not None:
                steady_sum += float(steady_score.mean())
            tracking_score = env.reward_terms.get("tracking")
            if tracking_score is not None:
                tracking_sum += float(tracking_score.mean())
            jerk_score = env.reward_terms.get("jerk")
            if jerk_score is not None:
                jerk_sum += float(jerk_score.mean())
            if progress_key == "ee_dist":
                grip_sum += float(env._gripper_opening().mean())

        rewards = torch.stack(rew_buf)
        values = torch.stack(val_buf)
        dones = torch.stack(done_buf)
        if algo == "grpo":
            returns = monte_carlo_returns(rewards, dones, args.gamma)
            advantages = returns
        else:
            advantages, returns = compute_gae(
                rewards,
                values,
                torch.stack(boot_buf),
                dones,
                args.gamma,
                args.lam,
            )

        def flat(items):
            return torch.cat(items, dim=0)

        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_obs = flat(obs_buf)
        flat_act = flat(act_buf)
        flat_logp = flat(logp_buf)
        flat_values = flat(val_buf)
        value_coef = args.value_coef
        normalize_advantages = True
        entropy_coef = float(args.entropy_coef)
        if algo == "grpo":
            group = int(args.group_size) if int(args.group_size) > 0 else int(args.num_envs)
            flat_advantages = stabilize_grpo_advantages(
                flat_returns,
                group,
                return_quantile=float(args.grpo_return_quantile),
                adv_clip=float(args.grpo_adv_clip),
            )
            value_coef = 0.0
            normalize_advantages = False
            # Critic-free updates amplify exploration noise; keep entropy cooler than PPO.
            entropy_coef = float(args.grpo_entropy_coef)
        if algo == "sym_ppo":
            # Walk: mirror only samples with a move/turn command (still-stand excluded).
            # Catch stores a predicted [x, y, z] in the command slot → full mirror.
            command_has_yaw = cfg.curriculum in LOCOMOTION_STAGES
            flat_obs, flat_act, flat_logp, flat_returns, flat_advantages, flat_values = augment_symmetric(
                flat_obs,
                flat_act,
                flat_logp,
                flat_returns,
                flat_advantages,
                flat_values,
                agent=agent,
                command_has_yaw=command_has_yaw,
            )
        metrics = ppo_update(
            agent,
            optimizer,
            flat_obs,
            flat_act,
            flat_logp,
            flat_returns,
            flat_advantages,
            clip=args.clip,
            epochs=args.epochs,
            minibatch_size=minibatch_size_for(args, flat_obs.shape[0]),
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=args.max_grad_norm,
            old_values=None if algo == "grpo" else flat_values,
            desired_kl=args.desired_kl if args.desired_kl > 0 else None,
            normalize_advantages=normalize_advantages,
        )
        elapsed = time.perf_counter() - started
        fps = args.num_envs * args.num_steps / max(elapsed, 1e-6)
        mean_reward = float(rewards.mean())
        writer.add_scalar("rollout/mean_reward", mean_reward, iteration)
        writer.add_scalar("rollout/fps", fps, iteration)
        if bc_teacher is not None and bc_imit_coef > 0.0:
            writer.add_scalar("rollout/bc_imit", bc_imit_sum / args.num_steps, iteration)
        for key, value in metrics.items():
            writer.add_scalar(f"train/{key}", value, iteration)
        if "episode_return" in env.extras:
            writer.add_scalar("rollout/episode_return", float(env.extras["episode_return"]), iteration)
            writer.add_scalar("rollout/success_rate", float(env.extras["success_rate"]), iteration)
        for name, value in env.reward_terms.items():
            writer.add_scalar(f"reward/{name}", float(value.mean()), iteration)
        grasp_rate = grasp_sum / args.num_steps
        mean_dist = dist_sum / args.num_steps
        writer.add_scalar("rollout/grasp_rate", grasp_rate, iteration)
        writer.add_scalar(f"rollout/{progress_key}", mean_dist, iteration)
        row = {
            "iteration": iteration,
            "reward": mean_reward,
            "grasp": grasp_rate,
            "fps": fps,
            "policy_loss": float(metrics["policy_loss"]),
            progress_key: mean_dist,
        }
        if "success_rate" in env.extras:
            row["success"] = float(env.extras["success_rate"])
        step_times.append(elapsed)
        remaining = (last_iteration - iteration) * (sum(step_times) / len(step_times))
        tracking_avg = tracking_sum / args.num_steps
        gait_avg = gait_sum / args.num_steps
        steady_avg = steady_sum / args.num_steps
        jerk_avg = jerk_sum / args.num_steps
        grip_avg = grip_sum / args.num_steps
        policy_loss = float(metrics["policy_loss"])
        # Color only metrics whose polarity matches task progress. gait/jerk/fps/policy
        # often move opposite to reward early on (stand still ⇒ gait↓, move more ⇒ jerk↑).
        prev = prev_log or {}
        pieces = [
            format_metric("reward", mean_reward, prev.get("reward"), higher_is_better=True, fmt=".3f"),
        ]
        if progress_key == "ee_dist":
            pieces.append(
                format_metric("ee_dist", mean_dist, prev.get("ee_dist"), higher_is_better=False, fmt=".3f")
            )
            pieces.append(
                format_metric("grasp", grasp_rate, prev.get("grasp"), higher_is_better=True, fmt=".2f")
            )
            pieces.append(f"grip {grip_avg:.3f}")
        elif progress_key == "approach":
            pieces.append(
                format_metric("approach", mean_dist, prev.get("approach"), higher_is_better=True, fmt=".3f")
            )
            pieces.append(
                format_metric("grasp", grasp_rate, prev.get("grasp"), higher_is_better=True, fmt=".2f")
            )
        else:
            pieces.append(
                format_metric("walk", mean_dist, prev.get("walk"), higher_is_better=True, fmt=".3f")
            )
            pieces.append(
                format_metric("tracking", tracking_avg, prev.get("tracking"), higher_is_better=True, fmt=".3f")
            )
            row["tracking"] = tracking_avg
            if cfg.curriculum in LOCOMOTION_STAGES:
                pieces.append(f"gait {gait_avg:.3f}")
                pieces.append(
                    format_metric("steady", steady_avg, prev.get("steady"), higher_is_better=True, fmt=".3f")
                )
                pieces.append(f"jerk {jerk_avg:.3f}")
                row["gait"] = gait_avg
                row["steady"] = steady_avg
                row["jerk"] = jerk_avg
        pieces.append(f"policy {policy_loss:.3f}")
        pieces.append(f"fps {fps:.0f}")
        row["kl"] = float(metrics["kl"])
        row["lr"] = float(metrics["lr"])
        row["entropy"] = float(metrics["entropy"])
        if float(metrics.get("aborted", 0.0)) > 0.0:
            pieces.append("update_aborted")
            row["aborted"] = 1
        append_metrics(args.log_dir / "metrics.jsonl", row)
        tracker.log_row(row)
        if cfg.curriculum in LOCOMOTION_STAGES:
            live_score = mean_dist
        elif cfg.curriculum == "catch" and success_count:
            live_score = success_sum / success_count
        else:
            live_score = None
        active = difficulty.update(env, live_score)
        detail = " ".join(pieces)
        if active:
            detail = f"{detail}  設定 {active}"
        print(
            f"iter {iteration}/{last_iteration}  残り {format_remaining(remaining)}  {detail}",
            flush=True,
        )
        prev_log = {
            "reward": mean_reward,
            "grasp": grasp_rate,
        }
        if progress_key == "ee_dist":
            prev_log["ee_dist"] = mean_dist
        elif progress_key == "approach":
            prev_log["approach"] = mean_dist
        else:
            prev_log["walk"] = mean_dist
            prev_log["tracking"] = tracking_avg
            if cfg.curriculum in LOCOMOTION_STAGES:
                prev_log["steady"] = steady_avg
        if iteration % args.save_interval == 0 or iteration == last_iteration:
            # Zero-pad to 6 so names sort in iteration order even as plain strings.
            saved = args.log_dir / f"ppo_{iteration:06d}.pt"
            save_checkpoint(
                saved,
                agent,
                optimizer,
                iteration,
                env.num_obs,
                env.num_actions,
                cfg.arm,
                cfg.curriculum,
                algo=normalize_algo(getattr(args, "algo", "ppo")),
            )
            tracker.log_checkpoint(saved)


def main() -> None:
    args = parse_args()
    error = ros_usage_error(enabled=args.ros, play=args.play, num_envs=args.num_envs)
    if error:
        raise SystemExit(error)
    algo = normalize_algo(getattr(args, "algo", "ppo"))
    if getattr(args, "easy_walk", False) and getattr(args, "curriculum", "carry") == "walk":
        # Keep a hard run's weights; easy walk always uses its own folder unless the user
        # already pointed --log-dir at a *_walk_easy path.
        if "walk_easy" not in Path(args.log_dir).name:
            args.log_dir = stage_log_dir("walk", algo=algo, easy=True)
            print(f"easy-walk: log-dir → {args.log_dir}", flush=True)
    prepare_record_path(args)
    cfg = make_config(args)
    missing = missing_play_checkpoint(args, cfg)
    if missing:
        raise SystemExit(missing)
    if args.viewer:
        print("Opening the 3D window. The first launch can take one to three minutes.", flush=True)
    env = PickAndCarryEnv(cfg=cfg)
    device = env.device
    init_std = initial_action_std(env.num_actions, args.leg_init_std, args.arm_init_std)
    agent = ActorCritic(env.num_obs, env.num_actions, init_std=init_std).to(device)
    if holds_arm_pose(cfg.curriculum):
        # Arm / gripper actions are not applied; keep them out of PPO entropy and KL.
        agent.set_action_mask(locomotion_action_mask(env.num_actions, device=device))
        print("action mask: legs only (arm dims ignored in policy loss)", flush=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    if not args.play:
        archived = archive_incompatible_logs(
            args.log_dir, env.num_obs, env.num_actions, cfg.arm, cfg.curriculum
        )
        if archived is not None:
            print(f"形の違う古い重みと記録を {archived} に移しました。", flush=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    obs = env.reset()
    env.save_fleet_snapshot()
    start_iteration, resumed_from = resume_agent(args, env, cfg, agent, optimizer)
    init_spec = getattr(args, "init_from", None)
    init_from = None
    warmed_from_imitation = False
    bc_teacher = None
    if resumed_from is None:
        # Auto-pick imitation warm-start in the easy-walk folder when no PPO checkpoint exists.
        # Prefer DAgger / BC for walk (env PPO), then ValueDICE / legacy GAIL.
        # Skip when the launcher explicitly asked for a true zero start (--no-auto-init-from).
        if (
            init_spec is None
            and getattr(args, "easy_walk", False)
            and not getattr(args, "no_auto_init_from", False)
        ):
            init_spec = "latest"
        if init_spec is not None:
            init_from = resolve_init_from(init_spec, Path(args.log_dir))
            peek = torch_load(init_from, env.device)
            # Footgun fix: picking a ppo_*.pt via --init-from used to wipe iteration to 0
            # and set warmed_from_imitation, overwriting earlier checkpoints.
            if is_rl_training_checkpoint(peek) or (
                init_from.name.startswith("ppo_") and not is_imitation_checkpoint(peek)
            ):
                start_iteration, resumed_from = load_rl_checkpoint_via_init_from(
                    init_from,
                    env,
                    cfg,
                    agent,
                    optimizer,
                    Path(args.log_dir),
                )
                warmed_from_imitation = False
            else:
                resumed_from = load_init_weights(
                    init_from,
                    env,
                    cfg,
                    agent,
                    reset_log_std=init_std,
                )
                start_iteration = 0
                warmed_from_imitation = True
    elif init_spec is not None:
        print(f"--init-from {init_spec} ignored because --resume loaded {resumed_from}", flush=True)
    bc_imit_coef = resolve_bc_imit_coef(args, warmed_from_imitation=warmed_from_imitation)
    if warmed_from_imitation and bc_imit_coef > 0.0:
        bc_teacher = freeze_actor_as_bc_teacher(agent.actor)
    gpu = ""
    if device.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        gpu = "  " + format_gpu_memory(free, total)
    print(
        f"{ALGO_LABELS[algo]} obs={env.num_obs} act={env.num_actions} envs={env.num_envs} "
        f"arm={cfg.arm} curriculum={cfg.curriculum} catch={cfg.catch_level} "
        f"control_split={cfg.control_split} wrist_camera={cfg.wrist_camera} "
        f"arm_disturb={cfg.arm_disturb} terrain={cfg.terrain} "
        f"urdf={env.urdf_path} device={device}{gpu}",
        flush=True,
    )
    print(
        f"同時に動かすロボットは {env.num_envs} 台です。状態は GPU 上に複製されるので、"
        "タスクマネージャーのメモリ欄はほとんど変わりません。",
        flush=True,
    )
    tracker = LocalTracker()
    if not args.play:
        info = {
            "algorithm": algo,
            "resumed_from": resumed_from,
            "start_iteration": start_iteration,
            "iterations_requested": args.iterations,
            "num_envs": env.num_envs,
            "num_steps": args.num_steps,
            "learning_rate": args.lr,
            "seed": args.seed,
            "obs_dim": env.num_obs,
            "act_dim": env.num_actions,
            "terrain": cfg.terrain,
            "terrain_amplitude_m": cfg.terrain_amplitude,
            "terrain_steps": cfg.terrain_steps,
            "arm": cfg.arm,
            "arm_disturb": cfg.arm_disturb,
            "push_robots": cfg.push_robots,
            "easy_walk": bool(getattr(args, "easy_walk", False)),
            "control_split": cfg.control_split,
            "wrist_camera": cfg.wrist_camera,
            "curriculum": cfg.curriculum,
            "catch_level": cfg.catch_level,
            "bc_imit_coef": bc_imit_coef,
            "warmed_from_imitation": warmed_from_imitation,
        }
        write_run(args.log_dir / "run.json", info)
        tracker.start(args.log_dir, info)
    try:
        if args.play:
            play_policy(args, cfg, env, agent, obs)
        else:
            run_training(
                args,
                cfg,
                env,
                agent,
                optimizer,
                writer,
                obs,
                start_iteration,
                tracker,
                bc_teacher=bc_teacher,
                bc_imit_coef=bc_imit_coef,
            )
    finally:
        tracker.close()
    writer.close()
    env.close()
    print(f"logs: {args.log_dir.resolve()}")
    if args.record:
        print(f"video: {Path(args.record).resolve()}")


if __name__ == "__main__":
    main()
