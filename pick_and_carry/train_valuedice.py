"""ValueDICE on the diagonal-trot teacher, then a PPO-compatible checkpoint.

Typical use (easy walk folder):

    python -m pick_and_carry.train_valuedice --easy-walk --seconds 90 --iterations 1000

Optional BC warm-start (recommended on a fresh run):

    python -m pick_and_carry.train_valuedice --easy-walk --bc-epochs 3000 --iterations 1000

Reuse a saved BC checkpoint (skips BC training by default):

    python -m pick_and_carry.train_valuedice --easy-walk --init-from latest --iterations 1000

Defaults mix walk env reward (``--env-coef 10``) with the ν imitation signal
and run ``--nu-epochs 1`` χ²-dual ν pass per PPO iteration.

Continue from an existing ``valuedice_trot.pt`` (skips BC by default):

    python -m pick_and_carry.train_valuedice --easy-walk --resume latest --iterations 1000

Then continue with task PPO:

    python -m pick_and_carry.train_ppo --curriculum walk --easy-walk --init-from \\
        logs/ppo_curriculum_walk_easy/valuedice_trot.pt --iterations 1000

``python -m pick_and_carry.train_gail`` is a compatibility alias for this module.
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
from pick_and_carry.console import format_metric
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
from pick_and_carry.valuedice import NuNet, apply_action_mask, train_nu


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Collect trot demos and train ValueDICE (PPO + ν network)."
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=60.0, help="Simulated seconds of teacher demos.")
    parser.add_argument("--iterations", type=int, default=1000, help="ValueDICE / PPO outer iterations.")
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=None,
        help="BC warm-start epochs before ValueDICE. Default 3000 on a fresh run, "
        "0 when --resume or --init-from. 0 skips.",
    )
    parser.add_argument("--bc-batch-size", type=int, default=4096)
    parser.add_argument(
        "--nu-epochs",
        type=int,
        default=None,
        help="ν network passes per iteration. Default 1 (or --disc-epochs if set).",
    )
    parser.add_argument(
        "--disc-epochs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # alias for --nu-epochs
    )
    parser.add_argument("--nu-batch-size", type=int, default=None)
    parser.add_argument("--disc-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--nu-lr", type=float, default=None)
    parser.add_argument("--disc-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--imit-coef",
        type=float,
        default=None,
        help="Weight on ν(s,a) reward. Default 1 (or --gail-coef if set).",
    )
    parser.add_argument("--gail-coef", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--env-coef",
        type=float,
        default=10.0,
        help="Weight on environment (walk) reward mixed into the imitation signal (0 = pure imitation).",
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
        help="PPO epochs per iteration.",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <log-dir>/valuedice_trot.pt",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Continue from checkpoint: 'latest' (= valuedice_trot.pt / gail_trot.pt) or a .pt path.",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Load policy weights only (e.g. bc_trot.pt) then start ValueDICE at iteration 0. "
        "'latest' / 'bc' = <log-dir>/bc_trot.pt. Ignored when --resume is set. "
        "Default --bc-epochs becomes 0 when this is set.",
    )
    parser.add_argument("--frequency", type=float, default=2.5)
    parser.add_argument("--demo-seconds", type=float, default=0.0)
    parser.add_argument("--play-seconds", type=float, default=None)
    parser.add_argument(
        "--demo-only",
        choices=("all", "teacher", "bc", "none"),
        default="none",
        help="Reuse train_bc demo recording; 'bc' loads valuedice_trot.pt / gail_trot.pt / bc_trot.pt.",
    )
    parser.add_argument(
        "--save-demos",
        choices=("all", "teacher", "bc", "none"),
        default="all",
        help="After training, which mp4s to write (bc = policy demos under valuedice_trot_*.mp4 names).",
    )
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args(argv)
    # Resolve aliases.
    args.nu_epochs = int(
        args.nu_epochs if args.nu_epochs is not None else (args.disc_epochs if args.disc_epochs is not None else 1)
    )
    args.nu_batch_size = int(
        args.nu_batch_size
        if args.nu_batch_size is not None
        else (args.disc_batch_size if args.disc_batch_size is not None else 4096)
    )
    args.nu_lr = float(args.nu_lr if args.nu_lr is not None else (args.disc_lr if args.disc_lr is not None else 3e-4))
    args.imit_coef = float(
        args.imit_coef if args.imit_coef is not None else (args.gail_coef if args.gail_coef is not None else 1.0)
    )
    # Keep legacy attribute names for any external callers.
    args.disc_epochs = args.nu_epochs
    args.disc_batch_size = args.nu_batch_size
    args.disc_lr = args.nu_lr
    args.gail_coef = args.imit_coef
    return args


def _resolve_resume_path(resume: str | None, log_dir: Path, output: Path) -> Path | None:
    if not resume:
        return None
    if resume in ("latest", "gail", "gail_trot", "valuedice", "valuedice_trot"):
        if output.is_file():
            return output
        for name in ("valuedice_trot.pt", "gail_trot.pt"):
            fallback = log_dir / name
            if fallback.is_file():
                return fallback
        raise SystemExit(f"--resume latest ですが ValueDICE/GAIL 重みがありません: {output}")
    path = Path(resume)
    if not path.is_file():
        raise SystemExit(f"--resume のファイルがありません: {path}")
    return path


def _resolve_init_from(init_from: str | None, log_dir: Path) -> Path | None:
    """Resolve ``--init-from`` to a policy checkpoint (typically ``bc_trot.pt``)."""
    if not init_from:
        return None
    if init_from in ("latest", "bc", "bc_trot"):
        path = log_dir / "bc_trot.pt"
        if not path.is_file():
            raise SystemExit(f"--init-from latest ですが BC 重みがありません: {path}")
        return path
    path = Path(init_from)
    if not path.is_file():
        raise SystemExit(f"--init-from のファイルがありません: {path}")
    return path


def _valuedice_iteration(blob: dict) -> int:
    if "valuedice_iteration" in blob:
        return int(blob["valuedice_iteration"])
    if "gail_iteration" in blob:
        return int(blob["gail_iteration"])
    return int(blob.get("iteration", 0) or 0)


# Alias used by train_gail / tests.
_gail_iteration = _valuedice_iteration


def _save_valuedice_blob(
    path: Path,
    *,
    agent: ActorCritic,
    nu_net: NuNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    obs_dim: int,
    act_dim: int,
    arm: str,
    algo: str,
    easy: bool,
) -> None:
    # Print the ValueDICE progress counter; rewrite ``iteration`` to 0 below so
    # ``train_ppo --init-from`` still starts task PPO from zero.
    save_checkpoint(
        path,
        agent,
        optimizer,
        iteration=int(iteration),
        obs_dim=obs_dim,
        act_dim=act_dim,
        arm=arm,
        curriculum="walk",
        algo=algo,
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["iteration"] = 0
    blob["valuedice_trot"] = True
    blob["gail_trot"] = True  # warm-start / play paths that still look for this flag
    blob["bc_trot"] = True
    blob["easy_walk"] = easy
    blob["valuedice_iteration"] = int(iteration)
    blob["gail_iteration"] = int(iteration)
    blob["nu_net"] = nu_net.state_dict()
    torch.save(blob, path)
    if iteration > 0:
        snap = path.parent / f"valuedice_{iteration:06d}.pt"
        torch.save(blob, snap)


def run_valuedice(
    *,
    env,
    agent: ActorCritic,
    nu_net: NuNet,
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
    opt_nu = torch.optim.Adam(nu_net.parameters(), lr=args.nu_lr)
    last_iteration = start_iteration + int(args.iterations)
    obs = env.reset()
    print(
        f"ValueDICE iterations {start_iteration + 1}…{last_iteration} "
        f"(+{args.iterations}) steps={args.num_steps} "
        f"imit_coef={args.imit_coef} env_coef={args.env_coef} "
        f"nu_epochs={args.nu_epochs} ppo_epochs={args.epochs}",
        flush=True,
    )
    finished = start_iteration
    prev_log: dict[str, float] | None = None
    for iteration in range(start_iteration + 1, last_iteration + 1):
        started = time.perf_counter()
        obs_buf, act_buf, next_obs_buf = [], [], []
        logp_buf, val_buf, rew_buf, boot_buf, done_buf = [], [], [], [], []
        nu_sum = 0.0
        env_sum = 0.0
        for _ in range(args.num_steps):
            with torch.no_grad():
                action, log_prob, value = agent.act(obs)
                next_obs, env_reward, done, info = env.step(action)
                masked_act = apply_action_mask(action, action_mask)
                nu_r = nu_net.valuedice_reward(obs, masked_act)
                reward = args.imit_coef * nu_r + args.env_coef * env_reward
                reward = reward + args.gamma * value * info["time_outs"]
                next_value = agent.value(next_obs)
                bootstrap = next_value * (~done).to(dtype=reward.dtype)
            obs_buf.append(obs)
            act_buf.append(action)
            next_obs_buf.append(next_obs)
            logp_buf.append(log_prob)
            val_buf.append(value)
            rew_buf.append(reward)
            boot_buf.append(bootstrap)
            done_buf.append(done.to(dtype=reward.dtype))
            nu_sum += float(nu_r.mean())
            env_sum += float(env_reward.mean())
            obs = next_obs

        flat_obs = torch.cat(obs_buf, dim=0)
        flat_act = torch.cat(act_buf, dim=0)
        flat_next = torch.cat(next_obs_buf, dim=0)
        with torch.no_grad():
            # Bootstrap action under the current policy at s' (ValueDICE residual).
            flat_next_act = agent.act_mean(flat_next)
        nu_metrics = train_nu(
            nu_net,
            opt_nu,
            expert_obs,
            expert_act,
            flat_obs.detach(),
            flat_act.detach(),
            next_obs=flat_next.detach(),
            next_act=flat_next_act.detach(),
            gamma=float(args.gamma),
            epochs=args.nu_epochs,
            batch_size=min(args.nu_batch_size, expert_obs.shape[0], flat_obs.shape[0]),
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
        r_nu = nu_sum / args.num_steps
        r_env = env_sum / args.num_steps
        r_mix = args.imit_coef * r_nu + args.env_coef * r_env
        if tracker is not None:
            tracker.log_row(
                {
                    "iteration": iteration,
                    "r_nu": r_nu,
                    "r_env": r_env,
                    "r_mix": r_mix,
                    "nu_loss": nu_metrics["nu_loss"],
                    "nu_expert": nu_metrics["nu_expert"],
                    "nu_policy": nu_metrics["nu_policy"],
                    "nu_delta": nu_metrics.get("nu_delta", 0.0),
                    "policy_loss": metrics["policy_loss"],
                }
            )
        span = max(1, args.iterations)
        if iteration == start_iteration + 1 or iteration % max(1, span // 20) == 0 or iteration == last_iteration:
            nu_loss = float(nu_metrics["nu_loss"])
            nu_e = float(nu_metrics["nu_expert"])
            nu_p = float(nu_metrics["nu_policy"])
            pi_loss = float(metrics["policy_loss"])
            prev = prev_log or {}
            print(
                f"  valuedice {iteration}/{last_iteration}  "
                f"{format_metric('r_nu', r_nu, prev.get('r_nu'), higher_is_better=True, fmt='.3f')}  "
                f"{format_metric('r_env', r_env, prev.get('r_env'), higher_is_better=True, fmt='.3f')}  "
                f"{format_metric('r_mix', r_mix, prev.get('r_mix'), higher_is_better=True, fmt='.3f')}  "
                f"{format_metric('nu_loss', nu_loss, prev.get('nu_loss'), higher_is_better=False, fmt='.3f')}  "
                f"nu_e {nu_e:.2f} nu_p {nu_p:.2f}  "
                f"pi_loss {pi_loss:.3f}  "
                f"{elapsed:.1f}s",
                flush=True,
            )
            prev_log = {
                "r_nu": r_nu,
                "r_env": r_env,
                "r_mix": r_mix,
                "nu_loss": nu_loss,
            }
        if save_path is not None and (
            iteration % max(1, args.save_interval) == 0 or iteration == last_iteration
        ):
            _save_valuedice_blob(
                save_path,
                agent=agent,
                nu_net=nu_net,
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
    out = args.output or (ns.log_dir / "valuedice_trot.pt")
    resume_path = _resolve_resume_path(args.resume, ns.log_dir, out)

    if args.demo_only != "none":
        which = _resolve_demo_which(args.demo_only, args.save_demos)
        if which is None:
            print("save-demos=none → デモ mp4 は書きません。", flush=True)
            return
        agent = None
        if which in ("all", "bc"):
            load_path = None
            for candidate in (out, ns.log_dir / "valuedice_trot.pt", ns.log_dir / "gail_trot.pt", ns.log_dir / "bc_trot.pt"):
                if candidate.is_file():
                    load_path = candidate
                    break
            if load_path is None:
                raise SystemExit(f"ValueDICE/BC デモ用の重みがありません: {out}")
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
    nu_net = NuNet(int(expert_obs.shape[-1]), int(expert_act.shape[-1])).to(device)

    start_iteration = 0
    init_path: Path | None = None
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
        if "nu_net" in blob:
            nu_net.load_state_dict(blob["nu_net"])
        elif "discriminator" in blob:
            print("旧 GAIL 識別器チェックポイントです。ν ネットワークは新規初期化します。", flush=True)
        else:
            print("checkpoint に ν が無いので、ν だけ新規です。", flush=True)
        start_iteration = _valuedice_iteration(blob)
        print(f"resumed ValueDICE from {resume_path} at iteration={start_iteration}", flush=True)
        if args.init_from:
            print(f"--init-from {args.init_from} は --resume があるため無視します。", flush=True)
    else:
        init_path = _resolve_init_from(args.init_from, ns.log_dir)
        if init_path is not None:
            blob = torch_load(init_path, device)
            if int(blob.get("obs_dim", -1)) != int(expert_obs.shape[-1]) or int(blob.get("act_dim", -1)) != int(
                expert_act.shape[-1]
            ):
                raise SystemExit(
                    f"--init-from {init_path} は obs/act "
                    f"{blob.get('obs_dim')}/{blob.get('act_dim')} で、"
                    f"今の環境は {expert_obs.shape[-1]}/{expert_act.shape[-1]} です。"
                )
            agent.load_state_dict(blob["model"])
            if blob.get("bc_trot"):
                kind = "BC"
            elif blob.get("valuedice_trot") or blob.get("gail_trot"):
                kind = "ValueDICE/GAIL policy"
            else:
                kind = "init"
            print(
                f"{kind} weights from {init_path.resolve()} "
                f"(ValueDICE starts at iteration 0, BC retrain skipped by default)",
                flush=True,
            )

    if args.bc_epochs is None:
        bc_epochs = 0 if (resume_path is not None or init_path is not None) else 3000
    else:
        bc_epochs = int(args.bc_epochs)

    finished = start_iteration
    tracker = LocalTracker()
    tracker.start(
        ns.log_dir,
        {
            "algorithm": "valuedice",
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
            "imit_coef": args.imit_coef,
            "env_coef": args.env_coef,
            "nu_epochs": args.nu_epochs,
            "demo_seconds": args.seconds,
            "resumed_from": str(resume_path) if resume_path is not None else None,
            "init_from": str(init_path) if init_path is not None else None,
        },
    )
    teacher_demos_done = False
    try:
        bc_path = ns.log_dir / "bc_trot.pt"
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
            _save_bc_warmstart_checkpoint(
                bc_path,
                agent=agent,
                obs_dim=int(expert_obs.shape[-1]),
                act_dim=int(expert_act.shape[-1]),
                arm=cfg.arm,
                algo=normalize_algo(ns.algo),
                easy=easy,
            )
            tracker.log_checkpoint(bc_path)
            print(f"saved BC warm-start checkpoint {bc_path.resolve()}", flush=True)

        # Mid-run videos of the BC policy (trained now, or loaded via --init-from).
        record_bc_demos = bc_epochs > 0 or init_path is not None
        mid_which = _resolve_demo_which("none", args.save_demos) if record_bc_demos else None
        if mid_which is not None:
            env.close()
            teacher_demos_done = _record_save_demos(
                cfg,
                args=args,
                log_dir=ns.log_dir,
                demo_seconds=demo_seconds,
                easy=easy,
                agent=agent,
                kind="bc",
                epochs=bc_epochs if bc_epochs > 0 else None,
                iteration=0,
                checkpoint_name=(init_path.name if init_path is not None and bc_epochs <= 0 else bc_path.name),
                rename_to_valuedice=False,
                include_teacher=True,
                include_policy=True,
            )
            env = PickAndCarryEnv(cfg=cfg)
            device = env.device
            expert_obs = expert_obs.to(device)
            expert_act = expert_act.to(device)
            agent = agent.to(device)
            nu_net = nu_net.to(device)

        finished = run_valuedice(
            env=env,
            agent=agent,
            nu_net=nu_net,
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

        optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
        _save_valuedice_blob(
            out,
            agent=agent,
            nu_net=nu_net,
            optimizer=optimizer,
            iteration=finished,
            obs_dim=int(expert_obs.shape[-1]),
            act_dim=int(expert_act.shape[-1]),
            arm=cfg.arm,
            algo=normalize_algo(ns.algo),
            easy=easy,
        )
        tracker.log_checkpoint(out)
        print(f"saved ValueDICE checkpoint {out.resolve()} (iteration={finished})", flush=True)
    finally:
        tracker.close()

    # Final demos: ValueDICE policy (and teacher only if BC phase did not already write it).
    _record_save_demos(
        cfg,
        args=args,
        log_dir=ns.log_dir,
        demo_seconds=demo_seconds,
        easy=easy,
        agent=agent,
        kind="valuedice",
        epochs=int(bc_epochs) if bc_epochs else None,
        iteration=int(finished),
        checkpoint_name=out.name,
        rename_to_valuedice=True,
        include_teacher=not teacher_demos_done,
        include_policy=True,
    )
    print(
        f"next: python -m pick_and_carry.train_ppo --curriculum walk --easy-walk "
        f"--init-from {out} --iterations 1000",
        flush=True,
    )
    print(
        f"追加学習: python -m pick_and_carry.train_valuedice --easy-walk --resume latest "
        f"--bc-epochs 0 --iterations 1000 --env-coef {args.env_coef} "
        f"--nu-epochs {args.nu_epochs} --log-dir {ns.log_dir}",
        flush=True,
    )


def _rename_policy_demos(folder: Path) -> None:
    """record_demos writes bc_trot_*; rename to valuedice_trot_* and alias gail_trot_*."""
    import shutil

    from pick_and_carry.trot_teacher import DEMO_SKILLS

    names = ["demo", *DEMO_SKILLS]
    for name in names:
        src = folder / f"bc_trot_{name}.mp4"
        valuedice = folder / f"valuedice_trot_{name}.mp4"
        gail = folder / f"gail_trot_{name}.mp4"
        if src.is_file():
            src.replace(valuedice)
        if valuedice.is_file() and not gail.is_file():
            shutil.copy2(valuedice, gail)


def _save_bc_warmstart_checkpoint(
    path: Path,
    *,
    agent: ActorCritic,
    obs_dim: int,
    act_dim: int,
    arm: str,
    algo: str,
    easy: bool,
) -> None:
    """Persist BC weights so mid-run demos match a reloadable checkpoint."""
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    save_checkpoint(
        path,
        agent,
        optimizer,
        iteration=0,
        obs_dim=obs_dim,
        act_dim=act_dim,
        arm=arm,
        curriculum="walk",
        algo=algo,
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["bc_trot"] = True
    blob["easy_walk"] = easy
    torch.save(blob, path)


def _record_save_demos(
    cfg,
    *,
    args,
    log_dir: Path,
    demo_seconds: float,
    easy: bool,
    agent: ActorCritic,
    kind: str,
    epochs: int | None,
    iteration: int,
    checkpoint_name: str,
    rename_to_valuedice: bool,
    include_teacher: bool,
    include_policy: bool,
) -> bool:
    """Write teacher/policy demos according to ``--save-demos``.

    Returns True if teacher demos were written (so a later phase can skip them).
    """
    from pick_and_carry.video_paths import make_video_run_dir, video_task_name

    which = _resolve_demo_which("none", args.save_demos)
    if which is None:
        print("save-demos=none → デモ mp4 はスキップします。", flush=True)
        return False

    want_teacher = include_teacher and which in ("all", "teacher")
    want_policy = include_policy and which in ("all", "bc")
    if not want_teacher and not want_policy:
        return False

    teacher_written = False
    videos: list[Path] = []
    if want_teacher:
        teacher_videos = record_demos(
            cfg,
            log_dir=log_dir,
            seconds=demo_seconds,
            frequency_hz=args.frequency,
            agent=None,
            which="teacher",
        )
        videos.extend(teacher_videos)
        teacher_written = True

    if want_policy:
        task = video_task_name("walk", easy=easy)
        video_dir = make_video_run_dir(
            log_dir,
            task=task,
            kind=kind,
            iteration=int(iteration),
            epochs=int(epochs) if epochs else None,
            extra={"checkpoint": str(checkpoint_name), "phase": kind},
        )
        cfg._demo_epochs = int(epochs) if epochs else 0
        policy_videos = record_demos(
            cfg,
            log_dir=log_dir,
            seconds=demo_seconds,
            frequency_hz=args.frequency,
            agent=agent.cpu(),
            which="bc",
            video_dir=video_dir,
        )
        if rename_to_valuedice:
            _rename_policy_demos(video_dir)
        videos.extend(policy_videos)

    for path in videos:
        print(f"demo ready: {path.resolve()}", flush=True)
    if videos:
        print(f"demo folder ({kind}): {videos[-1].parent.resolve()}", flush=True)
    return teacher_written


if __name__ == "__main__":
    main()
