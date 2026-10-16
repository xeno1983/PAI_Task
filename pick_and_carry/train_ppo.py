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
from pick_and_carry.algorithms import (
    ALGORITHMS,
    ALGO_LABELS,
    augment_symmetric,
    group_relative_advantages,
    monte_carlo_returns,
    normalize_algo,
)
from pick_and_carry.checkpoints import (
    archive_incompatible_logs,
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
)
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


def initial_action_std(num_actions: int, leg_std: float, arm_std: float, leg_count: int = 12):
    """Exploration noise per action: wide on the legs, narrow on the arm and gripper."""
    std = torch.full((num_actions,), float(arm_std))
    std[:leg_count] = float(leg_std)
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
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--num-steps", type=int, default=128)
    # Defaults follow rsl_rl / legged_gym: lr starts at 1e-3 and follows the KL target.
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--desired-kl", type=float, default=0.01, help="Adaptive learning rate target. 0 keeps --lr fixed.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=0, help="Samples per minibatch. 0 uses batch / --num-mini-batches.")
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--leg-init-std", type=float, default=0.3)
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
        help="Unitree Z1. Observation 81, action 19.",
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
    if cfg.curriculum in LOCOMOTION_STAGES:
        # Walk is the one stage on bumpy ground. The bumps start flat and grow with the score.
        cfg.terrain = True
        # The arm moves at random so the legs learn to balance it.
        cfg.arm_disturb = True
    elif cfg.curriculum != "carry":
        cfg.terrain = False
        cfg.terrain_steps = False
    if cfg.curriculum in ("haul", "reach", "catch"):
        cfg.catch_hold_steps = max(1, int(round(CATCH_HOLD_SECONDS / cfg.dt)))
    if cfg.curriculum in LOCOMOTION_STAGES and not args.play:
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


def run_training(args, cfg, env, agent, optimizer, writer, obs, start_iteration: int, tracker: LocalTracker) -> None:
    if start_iteration == 0:
        first = args.log_dir / "ppo_0000.pt"
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
    print(
        f"学習 {start_iteration}/{last_iteration} から開始します。1回終わるたびに残り時間を出します。",
        flush=True,
    )
    step_times: list[float] = []
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
        if algo == "grpo":
            group = int(args.group_size) if int(args.group_size) > 0 else int(args.num_envs)
            flat_advantages = group_relative_advantages(flat_returns, group)
            value_coef = 0.0
            normalize_advantages = False
        if algo == "sym_ppo":
            # Walk stores [vx, vy, yaw] in the command slot; catch stores a predicted [x, y, z].
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
            entropy_coef=args.entropy_coef,
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
        if progress_key == "ee_dist":
            detail = f"ee_dist {mean_dist:.3f} grasp {grasp_rate:.2f} grip {grip_sum / args.num_steps:.3f}"
        elif progress_key == "approach":
            detail = f"approach {mean_dist:.3f} grasp {grasp_rate:.2f}"
        else:
            detail = f"walk {mean_dist:.3f} tracking {tracking_sum / args.num_steps:.3f}"
            row["tracking"] = tracking_sum / args.num_steps
            if cfg.curriculum in LOCOMOTION_STAGES:
                detail = (
                    f"{detail} gait {gait_sum / args.num_steps:.3f} "
                    f"steady {steady_sum / args.num_steps:.3f} "
                    f"jerk {jerk_sum / args.num_steps:.3f}"
                )
                row["gait"] = gait_sum / args.num_steps
                row["steady"] = steady_sum / args.num_steps
                row["jerk"] = jerk_sum / args.num_steps
        row["kl"] = float(metrics["kl"])
        row["lr"] = float(metrics["lr"])
        row["entropy"] = float(metrics["entropy"])
        append_metrics(args.log_dir / "metrics.jsonl", row)
        tracker.log_row(row)
        if cfg.curriculum in LOCOMOTION_STAGES:
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
            saved = args.log_dir / f"ppo_{iteration:04d}.pt"
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
    gpu = ""
    if device.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        gpu = "  " + format_gpu_memory(free, total)
    algo = normalize_algo(getattr(args, "algo", "ppo"))
    print(
        f"{ALGO_LABELS[algo]} obs={env.num_obs} act={env.num_actions} envs={env.num_envs} "
        f"arm={cfg.arm} curriculum={cfg.curriculum} catch={cfg.catch_level} "
        f"control_split={cfg.control_split} wrist_camera={cfg.wrist_camera} "
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
            "control_split": cfg.control_split,
            "wrist_camera": cfg.wrist_camera,
            "curriculum": cfg.curriculum,
            "catch_level": cfg.catch_level,
        }
        write_run(args.log_dir / "run.json", info)
        tracker.start(args.log_dir, info)
    try:
        if args.play:
            play_policy(args, cfg, env, agent, obs)
        else:
            run_training(args, cfg, env, agent, optimizer, writer, obs, start_iteration, tracker)
    finally:
        tracker.close()
    writer.close()
    env.close()
    print(f"logs: {args.log_dir.resolve()}")
    if args.record:
        print(f"video: {Path(args.record).resolve()}")


if __name__ == "__main__":
    main()
