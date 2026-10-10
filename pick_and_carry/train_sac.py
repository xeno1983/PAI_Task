"""Off-policy SAC loop for the same pick-and-carry environment.

``step`` auto-resets, and the returned observation already belongs to the next
episode. Every done (timeout, fall, or success) is stored as terminal so the
backup does not evaluate Q on that reset observation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from pick_and_carry.agents import ReplayBuffer, SACAgent
from pick_and_carry.config import EnvConfig
from pick_and_carry.env import PickAndCarryEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train pick-and-carry with SAC on Genesis.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--control", choices=("position", "torque"), default="position")
    parser.add_argument("--no-assist-grasp", action="store_true")
    parser.add_argument("--freeze-legs", action="store_true", help="Stand still instead of walking.")
    parser.add_argument("--walk", action="store_true", help="Accepted for older commands. Walking is already the default.")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record", type=str, default=None, help="Save an mp4 of env 0 to this path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/sac"))
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig(
        num_envs=args.num_envs,
        backend=args.backend,
        show_viewer=args.viewer,
        record_path=args.record,
        control_mode=args.control,
        assist_grasp=not args.no_assist_grasp,
        freeze_legs=args.freeze_legs,
        seed=args.seed,
    )
    if args.viewer:
        print("Opening the 3D window. The first launch can take one to three minutes.", flush=True)
    env = PickAndCarryEnv(cfg=cfg)
    device = env.device
    agent = SACAgent(env.num_obs, env.num_actions, device, lr=args.lr, alpha=args.alpha, tau=args.tau)
    buffer = ReplayBuffer(args.buffer_size, env.num_obs, env.num_actions, device)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    obs = env.reset()
    print(f"SAC obs={env.num_obs} act={env.num_actions} envs={env.num_envs} urdf={env.urdf_path} device={device}")
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        with torch.no_grad():
            if step <= args.warmup:
                action = torch.empty((env.num_envs, env.num_actions), device=device).uniform_(-1.0, 1.0)
            else:
                action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, done.to(dtype=torch.float32))
        obs = next_obs

        metrics = None
        if buffer.size >= args.batch_size and step > args.warmup:
            for _ in range(args.updates_per_step):
                metrics = agent.update(buffer.sample(args.batch_size), args.gamma)

        if step % args.log_interval == 0:
            elapsed = time.perf_counter() - started
            fps = step * env.num_envs / max(elapsed, 1e-6)
            writer.add_scalar("rollout/mean_reward", float(reward.mean()), step)
            writer.add_scalar("rollout/fps", fps, step)
            if "episode_return" in info:
                writer.add_scalar("rollout/episode_return", float(info["episode_return"]), step)
                writer.add_scalar("rollout/success_rate", float(info["success_rate"]), step)
            for name, value in env.reward_terms.items():
                writer.add_scalar(f"reward/{name}", float(value.mean()), step)
            if metrics is not None:
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)
            print(f"step {step:05d} reward {float(reward.mean()):.3f} buffer {buffer.size} fps {fps:.0f}")
        if step % args.save_interval == 0 or step == args.steps:
            path = args.log_dir / f"sac_{step:05d}.pt"
            torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(), "step": step}, path)

    writer.close()
    env.close()
    print(f"logs: {args.log_dir.resolve()}")
    if args.record:
        print(f"video: {args.record}")


if __name__ == "__main__":
    main()
