"""Behavioral cloning of a diagonal-trot teacher, then save a PPO-compatible checkpoint.

Typical use (easy walk folder):

    python -m pick_and_carry.train_bc --easy-walk --seconds 90 --epochs 80

Then warm-start PPO:

    python -m pick_and_carry.train_ppo --curriculum walk --easy-walk --init-from \\
        logs/ppo_curriculum_walk_easy/bc_trot.pt --resume latest --iterations 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pick_and_carry.agents import ActorCritic
from pick_and_carry.algorithms import normalize_algo
from pick_and_carry.checkpoints import save_checkpoint
from pick_and_carry.curriculum import locomotion_action_mask, stage_log_dir
from pick_and_carry.train_ppo import initial_action_std, make_config
from pick_and_carry.trot_teacher import TrotTeacher


def parse_args():
    parser = argparse.ArgumentParser(description="Collect trot demos and train behavioral cloning.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=60.0, help="Simulated seconds of teacher rollouts.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=("z1",), default="z1")
    parser.add_argument("--algo", choices=("ppo", "grpo", "sym_ppo"), default="ppo")
    parser.add_argument("--easy-walk", action="store_true", default=True)
    parser.add_argument("--no-easy-walk", action="store_true", help="Collect on the hard walk setup.")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Checkpoint path. Default: <log-dir>/bc_trot.pt")
    parser.add_argument("--frequency", type=float, default=2.0, help="Trot frequency in Hz.")
    parser.add_argument("--play-seconds", type=float, default=0.0, help="If >0, record teacher mp4 after BC.")
    parser.add_argument("--record", type=str, default=None)
    return parser.parse_args()


def _build_walk_args(bc_args):
    """Reuse train_ppo config builder with a fake argparse namespace."""
    easy = bool(bc_args.easy_walk) and not bool(bc_args.no_easy_walk)
    algo = normalize_algo(bc_args.algo)
    log_dir = bc_args.log_dir
    if log_dir is None:
        log_dir = stage_log_dir("walk", algo=algo, easy=easy)
    ns = argparse.Namespace(
        num_envs=bc_args.num_envs,
        backend=bc_args.backend,
        viewer=False,
        record=bc_args.record,
        seed=bc_args.seed,
        terrain_steps=False,
        terrain_amplitude=0.0 if easy else 0.035,
        control_split="walk",
        arm=bc_args.arm,
        wrist_camera=False,
        object_xy=None,
        goal_xy=None,
        curriculum="walk",
        catch_level="static",
        easy_walk=easy,
        no_arm_disturb=True,
        play=False,
        ros=False,
        log_dir=log_dir,
        algo=algo,
        leg_init_std=0.3,
        arm_init_std=0.4,
    )
    return ns, easy


def collect_demos(env, teacher: TrotTeacher, seconds: float) -> tuple[torch.Tensor, torch.Tensor]:
    steps = max(1, int(round(seconds / env.cfg.dt)))
    obs_list: list[torch.Tensor] = []
    act_list: list[torch.Tensor] = []
    obs = env.reset()
    teacher.reset()
    print(f"collecting {steps} steps × {env.num_envs} envs (~{seconds:.0f}s)…", flush=True)
    started = time.perf_counter()
    for step in range(steps):
        action = teacher.act(env.command)
        obs_list.append(obs.detach())
        act_list.append(action.detach())
        obs, _, done, _ = env.step(action)
        if bool(done.any()):
            teacher.reset(torch.nonzero(done, as_tuple=False).squeeze(-1))
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"  collect {step + 1}/{steps}", flush=True)
    elapsed = time.perf_counter() - started
    obs_all = torch.cat(obs_list, dim=0)
    act_all = torch.cat(act_list, dim=0)
    print(f"collected {obs_all.shape[0]} transitions in {elapsed:.1f}s", flush=True)
    return obs_all, act_all


def train_bc(
    agent: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    action_mask: torch.Tensor,
) -> None:
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    n = obs.shape[0]
    mask = action_mask.to(dtype=obs.dtype, device=obs.device)
    print(f"BC epochs={epochs} batch={batch_size} samples={n}", flush=True)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, device=obs.device)
        total = 0.0
        count = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            pred = agent.actor.mean_action(obs[idx])
            err = (pred - actions[idx]).pow(2)
            loss = (err * mask.expand_as(err)).sum(dim=-1).mean() / mask.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"  bc epoch {epoch}/{epochs}  leg_mse {total / max(count, 1):.5f}", flush=True)


def main() -> None:
    args = parse_args()
    ns, easy = _build_walk_args(args)
    from pick_and_carry.env import PickAndCarryEnv

    cfg = make_config(ns)
    # BC collection should not push or disturb even if hard walk was requested with no-easy.
    if easy:
        cfg.terrain = False
        cfg.arm_disturb = False
        cfg.push_robots = False
        cfg.randomize_friction = False
    ns.log_dir.mkdir(parents=True, exist_ok=True)
    env = PickAndCarryEnv(cfg=cfg)
    device = env.device
    teacher = TrotTeacher(
        env.num_envs,
        dt=cfg.dt,
        frequency_hz=args.frequency,
        leg_action_scale=cfg.leg_action_scale,
        device=device,
    )
    obs, actions = collect_demos(env, teacher, args.seconds)

    init_std = initial_action_std(env.num_actions, ns.leg_init_std, ns.arm_init_std)
    agent = ActorCritic(env.num_obs, env.num_actions, init_std=init_std).to(device)
    mask = locomotion_action_mask(env.num_actions, device=device)
    agent.set_action_mask(mask)
    train_bc(
        agent,
        obs,
        actions,
        epochs=args.epochs,
        batch_size=min(args.batch_size, obs.shape[0]),
        lr=args.lr,
        action_mask=mask,
    )

    out = args.output or (ns.log_dir / "bc_trot.pt")
    # Optimizer state is empty Adam so PPO can replace it on --init-from.
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    save_checkpoint(
        out,
        agent,
        optimizer,
        iteration=0,
        obs_dim=env.num_obs,
        act_dim=env.num_actions,
        arm=cfg.arm,
        curriculum="walk",
        algo=normalize_algo(ns.algo),
    )
    # Also tag the file so resume logic can tell BC from PPO if needed.
    blob = torch.load(out, map_location="cpu", weights_only=False)
    blob["bc_trot"] = True
    blob["easy_walk"] = easy
    torch.save(blob, out)
    print(f"saved BC checkpoint {out.resolve()}", flush=True)

    if args.play_seconds and args.play_seconds > 0:
        record = args.record or str(ns.log_dir / "bc_trot_preview.mp4")
        print(f"preview teacher {args.play_seconds}s → {record}", flush=True)
        # Rebuild one env with recording.
        cfg.num_envs = 1
        cfg.record_path = record
        env.close()
        env = PickAndCarryEnv(cfg=cfg)
        teacher = TrotTeacher(1, dt=cfg.dt, frequency_hz=args.frequency, leg_action_scale=cfg.leg_action_scale, device=env.device)
        obs = env.reset()
        steps = max(1, int(round(args.play_seconds / cfg.dt)))
        for _ in range(steps):
            action = teacher.act(env.command)
            obs, _, done, _ = env.step(action)
            if bool(done.any()):
                teacher.reset(torch.nonzero(done, as_tuple=False).squeeze(-1))
        env.close()
        print(f"video: {Path(record).resolve()}", flush=True)
    else:
        env.close()
    print(f"next: python -m pick_and_carry.train_ppo --curriculum walk --easy-walk --init-from {out} --resume latest --iterations 1000")


if __name__ == "__main__":
    main()
