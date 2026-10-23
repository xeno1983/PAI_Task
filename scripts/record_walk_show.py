"""Record a closer chase-cam walk reel for the submission video.

Curriculum walk with the wide course camera looks tiny and collapsed. This
uses a flat course and a side camera that follows the base.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=ROOT / "logs" / "ppo_curriculum_walk" / "ppo_001900.pt")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--out", type=Path, default=LAUNCHER / "PickAndCarry_walk.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt.is_file():
        raise SystemExit(f"missing checkpoint: {args.ckpt}")

    from pick_and_carry.agents import ActorCritic
    from pick_and_carry.checkpoints import torch_load
    from pick_and_carry.config import EnvConfig
    from pick_and_carry.env import PickAndCarryEnv

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    nested_dir = ROOT / "logs" / "ppo_curriculum_walk" / "videos" / "walk_show"
    nested_dir.mkdir(parents=True, exist_ok=True)
    nested = nested_dir / f"chase_{args.ckpt.stem}.mp4"

    cfg = EnvConfig(
        num_envs=1,
        backend="gpu",
        show_viewer=False,
        record_path=str(nested),
        arm="z1",
        curriculum="walk",
        control_split="walk",
        terrain=False,
        terrain_steps=False,
        terrain_amplitude=0.0,
        arm_disturb=False,
        push_robots=False,
        randomize_friction=False,
        camera_res=(960, 540),
        # Side chase-cam (robot fills frame without clipping the arm).
        camera_pos=(0.55, -1.85, 0.88),
        camera_lookat=(0.75, 0.0, 0.32),
        camera_fov=45.0,
        episode_length_s=max(30.0, args.seconds + 5.0),
    )
    env = PickAndCarryEnv(cfg=cfg)
    blob = torch_load(args.ckpt, env.device)
    agent = ActorCritic(env.num_obs, env.num_actions, init_std=0.3).to(env.device)
    agent.load_state_dict(blob["model"])
    agent.eval()

    obs = env.reset()
    env.record_initial_pose(hold_s=0.6)
    steps = max(1, int(round(args.seconds / cfg.dt)))
    print(f"chase-cam walk {args.ckpt.name} for {args.seconds:.0f}s ({steps} steps)", flush=True)
    for _ in range(steps):
        with torch.no_grad():
            action = agent.act_mean(obs)
        obs, _, _, _ = env.step(action)
        if env.camera is not None:
            base = env.base_pos[0].detach().float().cpu()
            pos = (float(base[0] + 0.3), float(base[1] - 1.85), 0.88)
            look = (float(base[0] + 0.5), float(base[1]), 0.32)
            env.camera.set_pose(pos=list(pos), lookat=list(look))

    env.close()
    shutil.copy2(nested, out)
    print(f"saved {out} ({out.stat().st_size} bytes) from {nested}", flush=True)


if __name__ == "__main__":
    main()
