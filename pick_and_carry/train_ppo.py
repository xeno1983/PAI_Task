"""On-policy PPO loop for the Genesis pick-and-carry environment.

The environment auto-resets, so a timed-out transition bootstraps with V(s_t)
added into the reward and is then treated as terminal. Falls and successful
placements bootstrap with zero.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pick_and_carry.agents import ActorCritic, compute_gae, ppo_update
from pick_and_carry.config import EnvConfig, layout_dim
from pick_and_carry.curriculum import earlier_stages, later_stages, retreat_choice, stage_layouts, stage_log_dir
from pick_and_carry.env import PickAndCarryEnv
from pick_and_carry.ros_bridge import DEFAULT_HOST, DEFAULT_PORT, FileBridge, pack_rgb_image, ros_usage_error
from pick_and_carry.tracking import append_metrics, write_run


def format_remaining(seconds: float) -> str:
    """Predicted time left, as hours, minutes, and seconds."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}時間{minutes:02d}分{secs:02d}秒"
    return f"{minutes}分{secs:02d}秒"


def parse_args():
    parser = argparse.ArgumentParser(description="Train pick-and-carry with PPO on Genesis.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--control", choices=("position", "torque"), default="position")
    parser.add_argument("--no-assist-grasp", action="store_true")
    parser.add_argument("--freeze-legs", action="store_true", help="Stand still instead of walking.")
    parser.add_argument("--scripted-gait", action="store_true", help="Use the fixed trot instead of learning the legs.")
    parser.add_argument("--walk", action="store_true", help="Accepted for older commands. Walking is already the default.")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record", type=str, default=None, help="Save an mp4 of env 0 to this path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain-amplitude", type=float, default=None, help="Bump and step height in meters.")
    parser.add_argument("--terrain-steps", action="store_true", help="Add flat treads on the travel strips.")
    parser.add_argument(
        "--arm",
        choices=("z1",),
        default="z1",
        help="Unitree Z1. Observation 69, action 19.",
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
        choices=("carry", "walk", "jump", "stand", "sit", "rough", "haul", "catch"),
        default="carry",
        help="carry is pick-and-place. walk, then jump, stand, sit, then rough, haul, and catch.",
    )
    parser.add_argument(
        "--catch-level",
        choices=("static", "slow", "random", "noisy"),
        default="static",
        help="How the ball moves in the catch stage.",
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
    parser.add_argument("--play", action="store_true", help="Run the loaded policy and skip learning.")
    parser.add_argument("--seconds", type=float, default=20.0, help="Length of a --play recording, in simulation seconds.")
    parser.add_argument("--object-xy", type=str, default=None, help="Inference supply position, 'x,y' in meters.")
    parser.add_argument("--goal-xy", type=str, default=None, help="Inference drop position, 'x,y' in meters.")
    parser.add_argument(
        "--ros",
        action="store_true",
        help="Inference only. Publish robot state and accept a new drop point while playing.",
    )
    parser.add_argument("--ros-host", default=DEFAULT_HOST, help="Unused. Kept so older commands still start.")
    parser.add_argument("--ros-port", type=int, default=DEFAULT_PORT, help="Unused. Kept so older commands still start.")
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


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_matches(
    blob: dict,
    obs_dim: int,
    act_dim: int,
    scripted_gait: bool,
    arm: str,
    curriculum: str,
    *,
    match_arm: bool,
) -> bool:
    if int(blob.get("obs_dim", -1)) != obs_dim or int(blob.get("act_dim", -1)) != act_dim:
        return False
    # Files written before the Z1-only layout have no arm field and do not load.
    if match_arm and str(blob.get("arm", "panda")) != arm:
        return False
    # Files saved before the split used the fixed trot.
    if bool(blob.get("scripted_gait", True)) != scripted_gait:
        return False
    # Files saved before the curriculum field are pick-and-place.
    return str(blob.get("curriculum", "carry")) == curriculum


def latest_matching(
    log_dir: Path,
    obs_dim: int,
    act_dim: int,
    scripted_gait: bool,
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
    for path in sorted(log_dir.glob("ppo_*.pt")):
        blob = _torch_load(path, torch.device("cpu"))
        saved_arm = str(blob.get("arm", "panda"))
        if match_arm and saved_arm != arm and int(blob.get("obs_dim", -1)) == obs_dim:
            skipped_arm += 1
        if _checkpoint_matches(blob, obs_dim, act_dim, scripted_gait, arm, curriculum, match_arm=match_arm):
            compatible.append(path)
    if not compatible and skipped_arm:
        print(f"skipped {skipped_arm} checkpoint(s) from a different arm.", flush=True)
    return compatible[-1] if compatible else None


def resolve_checkpoint(
    log_dir: Path,
    resume: str,
    obs_dim: int,
    act_dim: int,
    scripted_gait: bool,
    arm: str,
    curriculum: str = "carry",
    catch_level: str = "static",
    root: Path | None = None,
    retreat: bool = False,
) -> Path | None:
    """Own folder first. Empty folders use an easier stage. ``retreat`` prefers a harder stage."""
    if resume != "latest":
        path = Path(resume)
        if not path.is_file():
            raise SystemExit(f"Checkpoint not found: {path}")
        return path
    # Old Panda checkpoints name a different arm and do not load.
    match_arm = True
    if retreat:
        for stage, level in later_stages(curriculum, catch_level):
            found = latest_matching(
                stage_log_dir(stage, level, arm, root=root),
                obs_dim,
                act_dim,
                scripted_gait,
                arm,
                stage,
                match_arm=match_arm,
            )
            if found is not None:
                return found
    own = latest_matching(log_dir, obs_dim, act_dim, scripted_gait, arm, curriculum)
    if own is not None:
        return own
    for stage, level in earlier_stages(curriculum, catch_level):
        found = latest_matching(
            stage_log_dir(stage, level, arm, root=root),
            obs_dim,
            act_dim,
            scripted_gait,
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
    found = resolve_checkpoint(
        args.log_dir,
        args.resume,
        layout_dim(observation),
        layout_dim(action),
        cfg.scripted_gait,
        cfg.arm,
        cfg.curriculum,
        cfg.catch_level,
        root=root,
        retreat=getattr(args, "retreat", False),
    )
    if found is not None:
        return None
    looked = [str(Path(args.log_dir))]
    for stage, level in earlier_stages(cfg.curriculum, cfg.catch_level):
        looked.append(str(stage_log_dir(stage, level, cfg.arm, root=root)))
    lines = [
        "再生できる重みがありません。",
        "探した場所: " + "、".join(looked),
        "先にメニューの 2 で、この段階を学習してください。",
    ]
    return "\n".join(lines)


def save_checkpoint(
    path: Path,
    agent,
    optimizer,
    iteration: int,
    obs_dim: int,
    act_dim: int,
    scripted_gait: bool,
    arm: str,
    curriculum: str = "carry",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "scripted_gait": scripted_gait,
            "arm": arm,
            "curriculum": curriculum,
        },
        path,
    )
    print(f"saved {path}", flush=True)


def make_config(args) -> EnvConfig:
    amplitude = 0.035 if args.terrain_amplitude is None else args.terrain_amplitude
    cfg = EnvConfig(
        num_envs=args.num_envs,
        backend=args.backend,
        show_viewer=args.viewer,
        record_path=args.record,
        control_mode=args.control,
        assist_grasp=not args.no_assist_grasp,
        freeze_legs=args.freeze_legs,
        scripted_gait=args.scripted_gait,
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
    if cfg.curriculum != "carry":
        cfg.scripted_gait = False
        cfg.control_split = "both"
    if cfg.curriculum in ("walk", "jump", "stand", "sit", "haul", "catch"):
        cfg.terrain = False
        cfg.terrain_steps = False
    elif cfg.curriculum == "rough":
        cfg.terrain = True
    if cfg.curriculum in ("haul", "catch"):
        cfg.catch_hold_steps = max(1, int(round(1.0 / cfg.dt)))
    if args.play and args.ros:
        # One click session should not snap the robot home on the 20 s training timeout.
        cfg.episode_length_s = max(cfg.episode_length_s, float(args.seconds))
        cfg.enable_camera = True
    return cfg


def _level_from_path(path: Path) -> str:
    name = path.parent.name
    prefix = "ppo_curriculum_catch_"
    if name.startswith(prefix) and name[len(prefix) :] in ("static", "slow", "random", "noisy"):
        return name[len(prefix) :]
    return "static"


def _stage_rank(stage: str, catch_level: str) -> int:
    """Higher means a harder task on the same ladder. Carry stays at zero."""
    if stage == "stand":
        return 1
    if stage == "sit":
        return 2
    if stage == "walk":
        return 3
    if stage == "jump":
        return 4
    if stage == "rough":
        return 5
    if stage == "haul":
        return 6
    if stage == "catch" and catch_level in ("static", "slow", "random", "noisy"):
        return 7 + ("static", "slow", "random", "noisy").index(catch_level)
    return 0


class _Difficulty:
    """Ease the live task when scores stay low, then climb back to the selected one."""

    def __init__(self, curriculum: str, catch_level: str):
        self.poor = 0
        self.good = 0
        self.kind = None
        self.current = 0
        self.target = 0
        self.labels: tuple[str, ...] = ()
        if curriculum == "rough":
            self.kind = "terrain"
            self.labels = ("平地", "半分の段差", "選んだ不整地")
            self.target = 2
            self.current = 2
        elif curriculum == "catch":
            self.kind = "catch"
            self.labels = ("静止", "ゆっくり", "ばらす", "ノイズ")
            self.target = ("static", "slow", "random", "noisy").index(catch_level)
            self.current = self.target

    def update(self, env, score: float | None) -> str:
        if self.kind is None:
            return ""
        new, self.poor, self.good = retreat_choice(
            score,
            current=self.current,
            target=self.target,
            poor_streak=self.poor,
            good_streak=self.good,
        )
        if new != self.current:
            previous = self.current
            self.current = new
            if self.kind == "terrain":
                env.set_terrain_mix((0.0, 0.5, 1.0)[new])
            else:
                env.set_catch_level(("static", "slow", "random", "noisy")[new])
            if new < previous:
                print(f"成績が低いので、{self.labels[new]} に戻しました。回復したら上げます。", flush=True)
            else:
                print(f"成績が戻ったので、{self.labels[new]} に上げました。", flush=True)
        if self.current == self.target:
            return ""
        return self.labels[self.current]


def resume_agent(args, env, cfg, agent, optimizer) -> tuple[int, str | None]:
    """Load a checkpoint when --resume is set. Returns the saved iteration and path."""
    if not args.resume:
        return 0, None
    ckpt_path = resolve_checkpoint(
        args.log_dir,
        args.resume,
        env.num_obs,
        env.num_actions,
        cfg.scripted_gait,
        cfg.arm,
        cfg.curriculum,
        cfg.catch_level,
        retreat=getattr(args, "retreat", False),
    )
    if ckpt_path is None:
        if args.play:
            raise SystemExit(
                "再生できる重みがありません。先にメニューの 2 で、この段階を学習してください。"
            )
        print(f"{args.log_dir.resolve()} にも前の段階にも重みがありません。最初から学習します。", flush=True)
        return 0, None
    blob = _torch_load(ckpt_path, env.device)
    obs_dim = int(blob.get("obs_dim", -1))
    act_dim = int(blob.get("act_dim", -1))
    saved_gait = bool(blob.get("scripted_gait", True))
    saved_arm = str(blob.get("arm", "panda"))
    saved_stage = str(blob.get("curriculum", "carry"))
    arm_required = True
    same_network = (
        obs_dim == env.num_obs
        and act_dim == env.num_actions
        and saved_gait == cfg.scripted_gait
        and (saved_arm == cfg.arm or not arm_required)
    )
    if not same_network:
        print(
            f"checkpoint {ckpt_path} is for arm {saved_arm} obs {obs_dim} act {act_dim} "
            f"scripted_gait {saved_gait}, this task is arm {cfg.arm} obs {env.num_obs} act {env.num_actions} "
            f"scripted_gait {cfg.scripted_gait}. Starting from scratch.",
            flush=True,
        )
        return 0, None
    agent.load_state_dict(blob["model"])
    same_folder = ckpt_path.parent.resolve() == Path(args.log_dir).resolve()
    if not same_folder:
        harder = _stage_rank(saved_stage, _level_from_path(ckpt_path)) > _stage_rank(cfg.curriculum, cfg.catch_level)
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


def run_training(args, cfg, env, agent, optimizer, writer, obs, start_iteration: int) -> None:
    if start_iteration == 0:
        save_checkpoint(
            args.log_dir / "ppo_0000.pt",
            agent,
            optimizer,
            0,
            env.num_obs,
            env.num_actions,
            cfg.scripted_gait,
            cfg.arm,
            cfg.curriculum,
        )
    last_iteration = start_iteration + args.iterations
    print(
        f"学習 {start_iteration}/{last_iteration} から開始します。1回終わるたびに残り時間を出します。",
        flush=True,
    )
    step_times: list[float] = []
    difficulty = _Difficulty(cfg.curriculum, cfg.catch_level)
    for iteration in range(start_iteration + 1, last_iteration + 1):
        started = time.perf_counter()
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, boot_buf, done_buf = [], [], [], [], [], [], []
        grasp_sum = 0.0
        dist_sum = 0.0
        grip_sum = 0.0
        success_sum = 0.0
        success_count = 0
        if cfg.curriculum in ("walk", "rough", "haul", "jump", "stand", "sit"):
            progress_key = "tracking"
        elif cfg.curriculum == "catch":
            progress_key = "approach"
        else:
            progress_key = "ee_dist"
        for _ in range(args.num_steps):
            with torch.no_grad():
                action, log_prob, value = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
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
            if progress_key == "ee_dist":
                grip_sum += float(env._gripper_opening().mean())

        rewards = torch.stack(rew_buf)
        values = torch.stack(val_buf)
        advantages, returns = compute_gae(
            rewards,
            values,
            torch.stack(boot_buf),
            torch.stack(done_buf),
            args.gamma,
            args.lam,
        )

        def flat(items):
            return torch.cat(items, dim=0)

        metrics = ppo_update(
            agent,
            optimizer,
            flat(obs_buf),
            flat(act_buf),
            flat(logp_buf),
            returns.reshape(-1),
            advantages.reshape(-1),
            clip=args.clip,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
        )
        elapsed = time.perf_counter() - started
        fps = args.num_envs * args.num_steps / max(elapsed, 1e-6)
        mean_reward = float(rewards.mean())
        writer.add_scalar("rollout/mean_reward", mean_reward, iteration)
        writer.add_scalar("rollout/fps", fps, iteration)
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
        append_metrics(args.log_dir / "metrics.jsonl", row)
        step_times.append(elapsed)
        remaining = (last_iteration - iteration) * (sum(step_times) / len(step_times))
        if progress_key == "ee_dist":
            detail = f"ee_dist {mean_dist:.3f} grasp {grasp_rate:.2f} grip {grip_sum / args.num_steps:.3f}"
        elif progress_key == "approach":
            detail = f"approach {mean_dist:.3f} grasp {grasp_rate:.2f}"
        else:
            detail = f"tracking {mean_dist:.3f}"
        if cfg.curriculum == "rough":
            live_score = mean_dist
        elif cfg.curriculum == "catch" and success_count:
            live_score = success_sum / success_count
        else:
            live_score = None
        active = difficulty.update(env, live_score)
        if active:
            detail = f"{detail}  設定 {active}"
        print(
            f"iter {iteration}/{last_iteration}  残り {format_remaining(remaining)}  "
            f"reward {mean_reward:.3f} {detail} "
            f"policy {metrics['policy_loss']:.3f} fps {fps:.0f}",
            flush=True,
        )
        if iteration % args.save_interval == 0 or iteration == last_iteration:
            save_checkpoint(
                args.log_dir / f"ppo_{iteration:04d}.pt",
                agent,
                optimizer,
                iteration,
                env.num_obs,
                env.num_actions,
                cfg.scripted_gait,
                cfg.arm,
                cfg.curriculum,
            )


def main() -> None:
    args = parse_args()
    error = ros_usage_error(enabled=args.ros, play=args.play, num_envs=args.num_envs)
    if error:
        raise SystemExit(error)
    cfg = make_config(args)
    missing = missing_play_checkpoint(args, cfg)
    if missing:
        raise SystemExit(missing)
    if args.viewer:
        print("Opening the 3D window. The first launch can take one to three minutes.", flush=True)
    env = PickAndCarryEnv(cfg=cfg)
    device = env.device
    agent = ActorCritic(env.num_obs, env.num_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    obs = env.reset()
    start_iteration, resumed_from = resume_agent(args, env, cfg, agent, optimizer)
    print(
        f"PPO obs={env.num_obs} act={env.num_actions} envs={env.num_envs} "
        f"arm={cfg.arm} curriculum={cfg.curriculum} catch={cfg.catch_level} "
        f"freeze_legs={cfg.freeze_legs} control_split={cfg.control_split} "
        f"wrist_camera={cfg.wrist_camera} urdf={env.urdf_path} device={device}",
        flush=True,
    )
    if not args.play:
        write_run(
            args.log_dir / "run.json",
            {
                "algorithm": "ppo",
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
                "freeze_legs": cfg.freeze_legs,
                "scripted_gait": cfg.scripted_gait,
                "arm": cfg.arm,
                "control_split": cfg.control_split,
                "wrist_camera": cfg.wrist_camera,
            },
        )
    if args.play:
        play_policy(args, cfg, env, agent, obs)
    else:
        run_training(args, cfg, env, agent, optimizer, writer, obs, start_iteration)
    writer.close()
    env.close()
    print(f"logs: {args.log_dir.resolve()}")
    if args.record:
        print(f"video: {Path(args.record).resolve()}")


if __name__ == "__main__":
    main()
