"""Behavioral cloning of a diagonal-trot teacher, then save a PPO-compatible checkpoint.

Typical use (easy walk folder):

    python -m pick_and_carry.train_bc --easy-walk --seconds 90 --epochs 80

Demos only (teacher / BC policy):

    python -m pick_and_carry.train_bc --demo-only teacher
    python -m pick_and_carry.train_bc --demo-only bc
    python -m pick_and_carry.train_bc --demo-only all

Then warm-start PPO:

    python -m pick_and_carry.train_ppo --curriculum walk --easy-walk --init-from \\
        logs/ppo_curriculum_walk_easy/bc_trot.pt --iterations 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pick_and_carry.agents import ActorCritic
from pick_and_carry.algorithms import normalize_algo
from pick_and_carry.checkpoints import save_checkpoint, torch_load
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
    parser.add_argument(
        "--demo-seconds",
        type=float,
        default=20.0,
        help="Record teacher/BC mp4s for this many seconds. 0 skips demos after training.",
    )
    parser.add_argument(
        "--play-seconds",
        type=float,
        default=None,
        help="Alias of --demo-seconds (kept for older scripts).",
    )
    parser.add_argument(
        "--demo-only",
        choices=("all", "teacher", "bc", "none"),
        default="none",
        help="Skip training; only record demos. 'bc' needs an existing bc_trot.pt.",
    )
    parser.add_argument("--record", type=str, default=None, help="Ignored; demos use fixed names under --log-dir.")
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


def _demo_cfg(base_cfg, log_dir: Path, record_path: Path):
    cfg = make_config(
        argparse.Namespace(
            num_envs=1,
            backend=base_cfg.backend,
            viewer=False,
            record=str(record_path),
            seed=base_cfg.seed,
            terrain_steps=False,
            terrain_amplitude=0.0,
            control_split="walk",
            arm=base_cfg.arm,
            wrist_camera=False,
            object_xy=None,
            goal_xy=None,
            curriculum="walk",
            catch_level="static",
            easy_walk=True,
            no_arm_disturb=True,
            play=False,
            ros=False,
            log_dir=log_dir,
        )
    )
    cfg.terrain = False
    cfg.arm_disturb = False
    cfg.push_robots = False
    cfg.randomize_friction = False
    cfg.num_envs = 1
    cfg.record_path = str(record_path)
    return cfg


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


def record_demos(
    cfg,
    *,
    log_dir: Path,
    seconds: float,
    frequency_hz: float,
    agent: ActorCritic | None,
    which: str = "all",
) -> list[Path]:
    """Record teacher and/or BC policy demos under ``log_dir``."""
    from pick_and_carry.env import PickAndCarryEnv

    paths: list[Path] = []
    if which in ("all", "teacher"):
        teacher_path = log_dir / "trot_teacher_demo.mp4"
        demo_cfg = _demo_cfg(cfg, log_dir, teacher_path)
        env = PickAndCarryEnv(cfg=demo_cfg)
        teacher = TrotTeacher(
            1,
            dt=demo_cfg.dt,
            frequency_hz=frequency_hz,
            leg_action_scale=demo_cfg.leg_action_scale,
            device=env.device,
        )
        obs = env.reset()
        steps = max(1, int(round(seconds / demo_cfg.dt)))
        # Force a clear +X command so the demo is a forward walk check, not a random mix.
        forward = torch.tensor([[0.40, 0.0, 0.0]], dtype=torch.float32, device=env.device)
        print(f"recording teacher {seconds:.0f}s (vx=+0.4) → {teacher_path}", flush=True)
        for _ in range(steps):
            env.command.copy_(forward)
            action = teacher.act(env.command)
            obs, _, done, _ = env.step(action)
            if bool(done.any()):
                teacher.reset(torch.nonzero(done, as_tuple=False).squeeze(-1))
                env.command.copy_(forward)
        env.close()
        print(f"video: {teacher_path.resolve()}", flush=True)
        paths.append(teacher_path)

    if which in ("all", "bc"):
        if agent is None:
            raise SystemExit("BC デモには学習済みの agent か bc_trot.pt が必要です。")
        bc_path = log_dir / "bc_trot_demo.mp4"
        demo_cfg = _demo_cfg(cfg, log_dir, bc_path)
        env = PickAndCarryEnv(cfg=demo_cfg)
        agent = agent.to(env.device)
        obs = env.reset()
        steps = max(1, int(round(seconds / demo_cfg.dt)))
        print(f"recording BC policy {seconds:.0f}s → {bc_path}", flush=True)
        with torch.no_grad():
            for _ in range(steps):
                action = agent.act_mean(obs)
                obs, _, _, _ = env.step(action)
        env.close()
        print(f"video: {bc_path.resolve()}", flush=True)
        paths.append(bc_path)
    return paths


def main() -> None:
    args = parse_args()
    ns, easy = _build_walk_args(args)
    from pick_and_carry.env import PickAndCarryEnv

    demo_seconds = args.demo_seconds if args.play_seconds is None else float(args.play_seconds)
    cfg = make_config(ns)
    if easy:
        cfg.terrain = False
        cfg.arm_disturb = False
        cfg.push_robots = False
        cfg.randomize_friction = False
    ns.log_dir.mkdir(parents=True, exist_ok=True)
    out = args.output or (ns.log_dir / "bc_trot.pt")

    if args.demo_only != "none":
        agent = None
        if args.demo_only in ("all", "bc"):
            if not out.is_file():
                raise SystemExit(f"BC デモ用の重みがありません: {out}")
            blob = torch_load(out, torch.device("cpu"))
            agent = ActorCritic(int(blob["obs_dim"]), int(blob["act_dim"]), init_std=0.3)
            agent.load_state_dict(blob["model"])
            agent.set_action_mask(locomotion_action_mask(int(blob["act_dim"])))
        videos = record_demos(
            cfg,
            log_dir=ns.log_dir,
            seconds=demo_seconds if demo_seconds > 0 else 20.0,
            frequency_hz=args.frequency,
            agent=agent,
            which="all" if args.demo_only == "all" else args.demo_only,
        )
        for path in videos:
            print(f"demo ready: {path.resolve()}", flush=True)
        return

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
    env.close()

    init_std = initial_action_std(int(actions.shape[-1]), ns.leg_init_std, ns.arm_init_std)
    agent = ActorCritic(int(obs.shape[-1]), int(actions.shape[-1]), init_std=init_std).to(device)
    mask = locomotion_action_mask(int(actions.shape[-1]), device=device)
    agent.set_action_mask(mask)
    train_bc(
        agent,
        obs.to(device),
        actions.to(device),
        epochs=args.epochs,
        batch_size=min(args.batch_size, obs.shape[0]),
        lr=args.lr,
        action_mask=mask,
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    save_checkpoint(
        out,
        agent,
        optimizer,
        iteration=0,
        obs_dim=int(obs.shape[-1]),
        act_dim=int(actions.shape[-1]),
        arm=cfg.arm,
        curriculum="walk",
        algo=normalize_algo(ns.algo),
    )
    blob = torch.load(out, map_location="cpu", weights_only=False)
    blob["bc_trot"] = True
    blob["easy_walk"] = easy
    torch.save(blob, out)
    print(f"saved BC checkpoint {out.resolve()}", flush=True)

    if demo_seconds and demo_seconds > 0:
        videos = record_demos(
            cfg,
            log_dir=ns.log_dir,
            seconds=demo_seconds,
            frequency_hz=args.frequency,
            agent=agent.cpu(),
            which="all",
        )
        for path in videos:
            print(f"demo ready: {path.resolve()}", flush=True)
    print(
        f"next: python -m pick_and_carry.train_ppo --curriculum walk --easy-walk "
        f"--init-from {out} --iterations 1000",
        flush=True,
    )


if __name__ == "__main__":
    main()
