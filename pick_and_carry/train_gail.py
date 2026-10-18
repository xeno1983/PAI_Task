"""GAIL on the diagonal-trot teacher, then a PPO-compatible checkpoint.

Typical use (easy walk folder):

    python -m pick_and_carry.train_gail --easy-walk --seconds 90 --iterations 1000

Optional BC warm-start (recommended on a fresh run):

    python -m pick_and_carry.train_gail --easy-walk --bc-epochs 3000 --iterations 1000

Defaults mix walk env reward (``--env-coef 10``) with the discriminator signal
and run ``--disc-epochs 4`` discriminator passes per PPO iteration.

Continue from an existing ``gail_trot.pt`` (skips BC by default):

    python -m pick_and_carry.train_gail --easy-walk --resume latest --iterations 1000

Then continue with task PPO:

    python -m pick_and_carry.train_ppo --curriculum walk --easy-walk --init-from \\
        logs/ppo_curriculum_walk_easy/gail_trot.pt --iterations 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pick_and_carry.agents import ActorCritic, compute_gae, ppo_update
from pick_and_carry.algorithms import normalize_algo
from pick_and_carry.checkpoints import save_checkpoint, torch_load
from pick_and_carry.curriculum import locomotion_action_mask
from pick_and_carry.gail import Discriminator, apply_action_mask, train_discriminator
from pick_and_carry.mlflow_log import LocalTracker
from pick_and_carry.train_bc import (
    _build_walk_args,
    _resolve_demo_which,
    collect_demos,
    record_demos,
    train_bc,
)
from pick_and_carry.train_ppo import initial_action_std, make_config, minibatch_size_for
from pick_and_carry.trot_teacher import TrotTeacher


def parse_args():
    parser = argparse.ArgumentParser(description="Collect trot demos and train GAIL (PPO + discriminator).")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=60.0, help="Simulated seconds of teacher demos.")
    parser.add_argument("--iterations", type=int, default=1000, help="GAIL / PPO outer iterations to run.")
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=None,
        help="BC warm-start epochs before GAIL. Default 3000 on a fresh run, 0 when --resume is set. 0 skips.",
    )
    parser.add_argument("--bc-batch-size", type=int, default=4096)
    parser.add_argument(
        "--disc-epochs",
        type=int,
        default=4,
        help="Discriminator passes per iteration (raised vs PPO so D stays ahead of the policy).",
    )
    parser.add_argument("--disc-batch-size", type=int, default=4096)
    parser.add_argument("--disc-lr", type=float, default=3e-4)
    parser.add_argument("--gail-coef", type=float, default=1.0, help="Weight on -log(1-D) reward.")
    parser.add_argument(
        "--env-coef",
        type=float,
        default=10.0,
        help="Weight on environment (walk) reward mixed into the GAIL signal (0 = pure imitation). "
        "Default 10 balances ~dt-scaled locomotion terms against softplus GAIL rewards.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="PPO epochs per iteration (matched near --disc-epochs so the discriminator keeps pace).",
    )
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=("z1",), default="z1")
    parser.add_argument("--algo", choices=("ppo", "grpo", "sym_ppo"), default="ppo")
    parser.add_argument("--easy-walk", action="store_true", default=True)
    parser.add_argument("--no-easy-walk", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Default: <log-dir>/gail_trot.pt")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Continue GAIL from a checkpoint: 'latest' (= <log-dir>/gail_trot.pt) or a .pt path.",
    )
    parser.add_argument("--frequency", type=float, default=2.5)
    parser.add_argument("--demo-seconds", type=float, default=0.0)
    parser.add_argument("--play-seconds", type=float, default=None)
    parser.add_argument(
        "--demo-only",
        choices=("all", "teacher", "bc", "none"),
        default="none",
        help="Reuse train_bc demo recording; 'bc' loads gail_trot.pt or bc_trot.pt.",
    )
    parser.add_argument(
        "--save-demos",
        choices=("all", "teacher", "bc", "none"),
        default="all",
        help="After training, which mp4s to write (bc = policy demos under gail_trot_*.mp4 names).",
    )
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=50)
    return parser.parse_args()


def _resolve_resume_path(resume: str | None, log_dir: Path, output: Path) -> Path | None:
    if not resume:
        return None
    if resume in ("latest", "gail", "gail_trot"):
        if output.is_file():
            return output
        fallback = log_dir / "gail_trot.pt"
        if fallback.is_file():
            return fallback
        raise SystemExit(f"--resume latest ですが GAIL 重みがありません: {output}")
    path = Path(resume)
    if not path.is_file():
        raise SystemExit(f"--resume のファイルがありません: {path}")
    return path


def _gail_iteration(blob: dict) -> int:
    if "gail_iteration" in blob:
        return int(blob["gail_iteration"])
    # Older files used iteration=0 for PPO init; treat missing as 0.
    return int(blob.get("iteration", 0) or 0)


def _save_gail_blob(
    path: Path,
    *,
    agent: ActorCritic,
    disc: Discriminator,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    obs_dim: int,
    act_dim: int,
    arm: str,
    algo: str,
    easy: bool,
) -> None:
    save_checkpoint(
        path,
        agent,
        optimizer,
        iteration=0,  # PPO --init-from always starts task PPO at 0
        obs_dim=obs_dim,
        act_dim=act_dim,
        arm=arm,
        curriculum="walk",
        algo=algo,
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["gail_trot"] = True
    blob["bc_trot"] = True
    blob["easy_walk"] = easy
    blob["gail_iteration"] = int(iteration)
    blob["discriminator"] = disc.state_dict()
    torch.save(blob, path)
    # Also keep a numbered snapshot for resume archaeology.
    if iteration > 0:
        snap = path.parent / f"gail_{iteration:06d}.pt"
        torch.save(blob, snap)


def run_gail(
    *,
    env,
    agent: ActorCritic,
    disc: Discriminator,
    expert_obs: torch.Tensor,
    expert_act: torch.Tensor,
    args,
    action_mask: torch.Tensor,
    start_iteration: int = 0,
    save_path: Path | None = None,
    easy: bool = True,
    arm: str = "z1",
    algo: str = "ppo",
    tracker: LocalTracker | None = None,
) -> int:
    device = env.device
    opt_pi = torch.optim.Adam(agent.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.disc_lr)
    last_iteration = start_iteration + int(args.iterations)
    obs = env.reset()
    print(
        f"GAIL iterations {start_iteration + 1}…{last_iteration} "
        f"(+{args.iterations}) steps={args.num_steps} "
        f"gail_coef={args.gail_coef} env_coef={args.env_coef} "
        f"disc_epochs={args.disc_epochs} ppo_epochs={args.epochs}",
        flush=True,
    )
    finished = start_iteration
    for iteration in range(start_iteration + 1, last_iteration + 1):
        started = time.perf_counter()
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, boot_buf, done_buf = [], [], [], [], [], [], []
        gail_sum = 0.0
        env_sum = 0.0
        for _ in range(args.num_steps):
            with torch.no_grad():
                action, log_prob, value = agent.act(obs)
                next_obs, env_reward, done, info = env.step(action)
                masked_act = apply_action_mask(action, action_mask)
                gail_r = disc.gail_reward(obs, masked_act)
                reward = args.gail_coef * gail_r + args.env_coef * env_reward
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
            gail_sum += float(gail_r.mean())
            env_sum += float(env_reward.mean())
            obs = next_obs

        flat_obs = torch.cat(obs_buf, dim=0)
        flat_act = torch.cat(act_buf, dim=0)
        disc_metrics = train_discriminator(
            disc,
            opt_d,
            expert_obs,
            expert_act,
            flat_obs.detach(),
            flat_act.detach(),
            epochs=args.disc_epochs,
            batch_size=min(args.disc_batch_size, expert_obs.shape[0], flat_obs.shape[0]),
            action_mask=action_mask,
        )

        rewards = torch.stack(rew_buf)
        values = torch.stack(val_buf)
        dones = torch.stack(done_buf)
        advantages, returns = compute_gae(
            rewards,
            values,
            torch.stack(boot_buf),
            dones,
            args.gamma,
            args.lam,
        )
        metrics = ppo_update(
            agent,
            opt_pi,
            flat_obs,
            flat_act,
            torch.cat(logp_buf, dim=0),
            returns.reshape(-1),
            advantages.reshape(-1),
            clip=args.clip,
            epochs=args.epochs,
            minibatch_size=minibatch_size_for(args, flat_obs.shape[0]),
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            old_values=values.reshape(-1),
            desired_kl=args.desired_kl if args.desired_kl > 0 else None,
        )
        elapsed = time.perf_counter() - started
        finished = iteration
        r_gail = gail_sum / args.num_steps
        r_env = env_sum / args.num_steps
        r_mix = args.gail_coef * r_gail + args.env_coef * r_env
        if tracker is not None:
            tracker.log_row(
                {
                    "iteration": iteration,
                    "r_gail": r_gail,
                    "r_env": r_env,
                    "r_mix": r_mix,
                    "disc_loss": disc_metrics["disc_loss"],
                    "disc_acc": disc_metrics["disc_acc"],
                    "policy_loss": metrics["policy_loss"],
                }
            )
        span = max(1, args.iterations)
        if iteration == start_iteration + 1 or iteration % max(1, span // 20) == 0 or iteration == last_iteration:
            print(
                f"  gail {iteration}/{last_iteration}  "
                f"r_gail {r_gail:.3f}  "
                f"r_env {r_env:.3f}  "
                f"r_mix {r_mix:.3f}  "
                f"disc_loss {disc_metrics['disc_loss']:.3f}  "
                f"disc_acc {disc_metrics['disc_acc']:.2f}  "
                f"pi_loss {metrics['policy_loss']:.3f}  "
                f"{elapsed:.1f}s",
                flush=True,
            )
        if save_path is not None and (
            iteration % max(1, args.save_interval) == 0 or iteration == last_iteration
        ):
            _save_gail_blob(
                save_path,
                agent=agent,
                disc=disc,
                optimizer=opt_pi,
                iteration=iteration,
                obs_dim=int(expert_obs.shape[-1]),
                act_dim=int(expert_act.shape[-1]),
                arm=arm,
                algo=algo,
                easy=easy,
            )
            if tracker is not None:
                tracker.log_checkpoint(save_path)
    return finished


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
    out = args.output or (ns.log_dir / "gail_trot.pt")
    resume_path = _resolve_resume_path(args.resume, ns.log_dir, out)

    if args.demo_only != "none":
        which = _resolve_demo_which(args.demo_only, args.save_demos)
        if which is None:
            print("save-demos=none → デモ mp4 は書きません。", flush=True)
            return
        agent = None
        if which in ("all", "bc"):
            load_path = out if out.is_file() else (ns.log_dir / "bc_trot.pt")
            if not load_path.is_file():
                raise SystemExit(f"GAIL/BC デモ用の重みがありません: {load_path}")
            blob = torch_load(load_path, torch.device("cpu"))
            agent = ActorCritic(int(blob["obs_dim"]), int(blob["act_dim"]), init_std=0.3)
            agent.load_state_dict(blob["model"])
            agent.set_action_mask(locomotion_action_mask(int(blob["act_dim"])))
        videos = record_demos(
            cfg,
            log_dir=ns.log_dir,
            seconds=demo_seconds,
            frequency_hz=args.frequency,
            agent=agent,
            which=which,
        )
        if which in ("all", "bc") and agent is not None and videos:
            _rename_policy_demos(videos[0].parent)
        for path in videos:
            print(f"demo ready: {path.resolve()}", flush=True)
        if videos:
            print(f"demo folder: {videos[0].parent.resolve()}", flush=True)
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
    expert_obs, expert_act = collect_demos(env, teacher, args.seconds)
    env.close()
    env = PickAndCarryEnv(cfg=cfg)
    device = env.device
    expert_obs = expert_obs.to(device)
    expert_act = expert_act.to(device)

    init_std = initial_action_std(int(expert_act.shape[-1]), ns.leg_init_std, ns.arm_init_std)
    agent = ActorCritic(int(expert_obs.shape[-1]), int(expert_act.shape[-1]), init_std=init_std).to(device)
    mask = locomotion_action_mask(int(expert_act.shape[-1]), device=device)
    agent.set_action_mask(mask)
    disc = Discriminator(int(expert_obs.shape[-1]), int(expert_act.shape[-1])).to(device)

    start_iteration = 0
    if resume_path is not None:
        blob = torch_load(resume_path, device)
        if int(blob.get("obs_dim", -1)) != int(expert_obs.shape[-1]) or int(blob.get("act_dim", -1)) != int(
            expert_act.shape[-1]
        ):
            raise SystemExit(
                f"--resume {resume_path} は obs/act "
                f"{blob.get('obs_dim')}/{blob.get('act_dim')} で、"
                f"今の環境は {expert_obs.shape[-1]}/{expert_act.shape[-1]} です。"
            )
        agent.load_state_dict(blob["model"])
        if "discriminator" in blob:
            disc.load_state_dict(blob["discriminator"])
        else:
            print("checkpoint に識別器が無いので、識別器だけ新規です。", flush=True)
        start_iteration = _gail_iteration(blob)
        print(f"resumed GAIL from {resume_path} at gail_iteration={start_iteration}", flush=True)

    if args.bc_epochs is None:
        bc_epochs = 0 if resume_path is not None else 3000
    else:
        bc_epochs = int(args.bc_epochs)

    finished = start_iteration
    tracker = LocalTracker()
    tracker.start(
        ns.log_dir,
        {
            "algorithm": "gail",
            "curriculum": "walk",
            "arm": cfg.arm,
            "obs_dim": int(expert_obs.shape[-1]),
            "act_dim": int(expert_act.shape[-1]),
            "num_envs": cfg.num_envs,
            "num_steps": args.num_steps,
            "iterations_requested": args.iterations,
            "start_iteration": start_iteration,
            "bc_epochs": bc_epochs,
            "learning_rate": args.lr,
            "seed": args.seed,
            "easy_walk": easy,
            "gail_coef": args.gail_coef,
            "env_coef": args.env_coef,
            "disc_epochs": args.disc_epochs,
            "demo_seconds": args.seconds,
            "resumed_from": str(resume_path) if resume_path is not None else None,
        },
    )
    try:
        if bc_epochs > 0:
            print(f"BC warm-start epochs={bc_epochs}…", flush=True)
            train_bc(
                agent,
                expert_obs,
                expert_act,
                epochs=bc_epochs,
                batch_size=min(args.bc_batch_size, expert_obs.shape[0]),
                lr=3e-4,
                action_mask=mask,
                tracker=tracker,
            )

        finished = run_gail(
            env=env,
            agent=agent,
            disc=disc,
            expert_obs=expert_obs,
            expert_act=expert_act,
            args=args,
            action_mask=mask,
            start_iteration=start_iteration,
            save_path=out,
            easy=easy,
            arm=cfg.arm,
            algo=normalize_algo(ns.algo),
            tracker=tracker,
        )
        env.close()

        # Final save (also covered by last-iter save inside run_gail, but keep explicit).
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
        _save_gail_blob(
            out,
            agent=agent,
            disc=disc,
            optimizer=optimizer,
            iteration=finished,
            obs_dim=int(expert_obs.shape[-1]),
            act_dim=int(expert_act.shape[-1]),
            arm=cfg.arm,
            algo=normalize_algo(ns.algo),
            easy=easy,
        )
        tracker.log_checkpoint(out)
        print(f"saved GAIL checkpoint {out.resolve()} (gail_iteration={finished})", flush=True)
    finally:
        tracker.close()

    which = _resolve_demo_which("none", args.save_demos)
    if which is None:
        print("save-demos=none → デモ mp4 はスキップします。", flush=True)
    else:
        from pick_and_carry.video_paths import make_video_run_dir, video_task_name

        task = video_task_name("walk", easy=easy)
        video_dir = make_video_run_dir(
            ns.log_dir,
            task=task,
            kind="gail",
            iteration=int(finished),
            epochs=int(bc_epochs) if bc_epochs else None,
            extra={"checkpoint": str(out.name)},
        )
        if which == "all":
            teacher_videos = record_demos(
                cfg,
                log_dir=ns.log_dir,
                seconds=demo_seconds,
                frequency_hz=args.frequency,
                agent=None,
                which="teacher",
            )
            cfg._demo_epochs = int(bc_epochs)
            policy_videos = record_demos(
                cfg,
                log_dir=ns.log_dir,
                seconds=demo_seconds,
                frequency_hz=args.frequency,
                agent=agent.cpu(),
                which="bc",
                video_dir=video_dir,
            )
            _rename_policy_demos(video_dir)
            videos = teacher_videos + policy_videos
        else:
            videos = record_demos(
                cfg,
                log_dir=ns.log_dir,
                seconds=demo_seconds,
                frequency_hz=args.frequency,
                agent=agent.cpu(),
                which=which,
                video_dir=video_dir if which == "bc" else None,
            )
            if which == "bc":
                _rename_policy_demos(video_dir)
        for path in videos:
            print(f"demo ready: {path.resolve()}", flush=True)
        if videos:
            print(f"demo folder: {videos[-1].parent.resolve()}", flush=True)
    print(
        f"next: python -m pick_and_carry.train_ppo --curriculum walk --easy-walk "
        f"--init-from {out} --iterations 1000",
        flush=True,
    )
    print(
        f"追加学習: python -m pick_and_carry.train_gail --easy-walk --resume latest "
        f"--bc-epochs 0 --iterations 1000 --env-coef {args.env_coef} "
        f"--disc-epochs {args.disc_epochs} --log-dir {ns.log_dir}",
        flush=True,
    )


def _rename_policy_demos(folder: Path) -> None:
    """record_demos writes bc_trot_*; rename to gail_trot_* for this entry point."""
    mapping = [("bc_trot_demo.mp4", "gail_trot_demo.mp4")]
    from pick_and_carry.trot_teacher import DEMO_SKILLS

    for skill in DEMO_SKILLS:
        mapping.append((f"bc_trot_{skill}.mp4", f"gail_trot_{skill}.mp4"))
    for src_name, dst_name in mapping:
        src = folder / src_name
        dst = folder / dst_name
        if src.is_file():
            src.replace(dst)


if __name__ == "__main__":
    main()
