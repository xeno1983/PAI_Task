"""DAgger (classic / gated) on the diagonal-trot teacher, then a PPO-compatible checkpoint.

Typical use (easy walk folder):

    python -m pick_and_carry.train_dagger --easy-walk --rounds 8 --seconds 30

Reuse a saved BC policy as the student start:

    python -m pick_and_carry.train_dagger --easy-walk --init-from latest --rounds 8

Gated mode only stores labels where policy and teacher disagree:

    python -m pick_and_carry.train_dagger --easy-walk --mode gated --threshold 0.15

Then continue with task PPO:

    python -m pick_and_carry.train_ppo --curriculum walk --easy-walk --init-from \\
        logs/ppo_curriculum_walk_easy/dagger_trot.pt --iterations 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pick_and_carry.agents import ActorCritic
from pick_and_carry.algorithms import normalize_algo
from pick_and_carry.checkpoints import save_checkpoint, torch_load
from pick_and_carry.curriculum import locomotion_action_mask
from pick_and_carry.console import format_metric
from pick_and_carry.dagger import action_disagreement, dagger_beta, should_query_teacher
from pick_and_carry.mlflow_log import LocalTracker
from pick_and_carry.train_bc import (
    _build_walk_args,
    _resolve_demo_which,
    collect_demos,
    record_demos,
    train_bc,
)
from pick_and_carry.train_ppo import initial_action_std, make_config
from pick_and_carry.trot_teacher import TrotTeacher


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="DAgger interactive imitation with the trot teacher (classic or gated)."
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Simulated seconds of rollouts per DAgger round.",
    )
    parser.add_argument(
        "--seed-seconds",
        type=float,
        default=60.0,
        help="Initial teacher-only demos before round 1 (0 skips; ignored when --init-from loads BC).",
    )
    parser.add_argument("--rounds", type=int, default=8, help="DAgger aggregation rounds.")
    parser.add_argument("--bc-epochs", type=int, default=200, help="BC epochs after each round.")
    parser.add_argument("--seed-bc-epochs", type=int, default=1000, help="BC epochs on seed demos (0 if --init-from).")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--mode",
        choices=("classic", "gated"),
        default="classic",
        help="classic = label every state; gated = label only when disagreement >= --threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Gated disagreement threshold (masked L2 on actions).",
    )
    parser.add_argument("--beta0", type=float, default=1.0, help="Initial expert mix probability for env steps.")
    parser.add_argument("--beta-decay", type=float, default=0.85, help="Per-round beta multiplier.")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Load student weights (e.g. bc_trot.pt). 'latest'/'bc' = <log-dir>/bc_trot.pt.",
    )
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
        help="Default: <log-dir>/dagger_trot.pt",
    )
    parser.add_argument("--frequency", type=float, default=2.5)
    parser.add_argument("--demo-seconds", type=float, default=0.0)
    parser.add_argument("--play-seconds", type=float, default=None)
    parser.add_argument(
        "--demo-only",
        choices=("all", "teacher", "bc", "none"),
        default="none",
        help="Reuse train_bc demo recording; 'bc' loads dagger_trot.pt / bc_trot.pt.",
    )
    parser.add_argument(
        "--save-demos",
        choices=("all", "teacher", "bc", "none"),
        default="all",
        help="After training, which mp4s to write (bc = student policy demos).",
    )
    parser.add_argument("--record", type=str, default=None)
    return parser.parse_args(argv)


def _resolve_init_from(init_from: str | None, log_dir: Path) -> Path | None:
    if not init_from:
        return None
    if init_from in ("latest", "dagger", "dagger_trot"):
        for name in ("dagger_trot.pt", "bc_trot.pt"):
            path = log_dir / name
            if path.is_file():
                return path
        raise SystemExit(f"--init-from latest ですが DAgger/BC 重みがありません: {log_dir}")
    if init_from in ("bc", "bc_trot"):
        path = log_dir / "bc_trot.pt"
        if path.is_file():
            return path
        raise SystemExit(f"--init-from bc ですが BC 重みがありません: {path}")
    path = Path(init_from)
    if not path.is_file():
        raise SystemExit(f"--init-from のファイルがありません: {path}")
    return path


def _save_dagger_checkpoint(
    path: Path,
    *,
    agent: ActorCritic,
    obs_dim: int,
    act_dim: int,
    arm: str,
    algo: str,
    easy: bool,
    mode: str,
    round_idx: int,
) -> None:
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
    blob["dagger_trot"] = True
    blob["bc_trot"] = True  # play / warm-start paths that look for BC-compatible flags
    blob["easy_walk"] = easy
    blob["dagger_mode"] = mode
    blob["dagger_round"] = int(round_idx)
    torch.save(blob, path)


def collect_dagger_round(
    env,
    teacher: TrotTeacher,
    agent: ActorCritic,
    *,
    seconds: float,
    beta: float,
    mode: str,
    threshold: float,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """One DAgger round: student visits states; labels are always teacher actions."""
    steps = max(1, int(round(seconds / env.cfg.dt)))
    resample_every = max(1, int(round(2.5 / env.cfg.dt)))
    obs_list: list[torch.Tensor] = []
    act_list: list[torch.Tensor] = []
    query_count = 0
    visit_count = 0
    disagree_sum = 0.0
    obs = env.reset()
    teacher.reset()
    teacher.sample_skills()
    obs = env.set_commands(teacher.commands_for_skills(), hold=True)
    print(
        f"  dagger rollout steps={steps} beta={beta:.3f} mode={mode} "
        f"threshold={threshold:.3f}",
        flush=True,
    )
    started = time.perf_counter()
    for step in range(steps):
        if step > 0 and step % resample_every == 0:
            teacher.sample_skills()
            obs = env.set_commands(teacher.commands_for_skills(), hold=True)
        with torch.no_grad():
            # Peek teacher action, then advance teacher phase only where the expert
            # action is actually applied (β-mix). Otherwise CPG timers drift and
            # later labels no longer match student-visited states.
            teacher_act = teacher.act(env.command, advance=False)
            policy_act = agent.actor.mean_action(obs)
            disagree = action_disagreement(policy_act, teacher_act, action_mask)
            query = should_query_teacher(disagree, threshold=threshold, mode=mode)
            use_expert = torch.rand(env.num_envs, device=obs.device) < float(beta)
            step_act = torch.where(use_expert.unsqueeze(-1), teacher_act, policy_act)
            if bool(use_expert.any()):
                snap = teacher.snapshot_state()
                _ = teacher.act(env.command, advance=True)
                teacher.restore_state(snap, env_mask=~use_expert)
        visit_count += int(env.num_envs)
        disagree_sum += float(disagree.mean())
        if bool(query.any()):
            q_ids = torch.nonzero(query, as_tuple=False).squeeze(-1)
            obs_list.append(obs[q_ids].detach())
            act_list.append(teacher_act[q_ids].detach())
            query_count += int(q_ids.numel())
        obs, _, done, _ = env.step(step_act)
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            teacher.reset(done_ids)
            teacher.sample_skills()
            obs = env.set_commands(teacher.commands_for_skills(), hold=True)
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    step {step + 1}/{steps}  queries={query_count}", flush=True)
    env.hold_commands = False
    elapsed = time.perf_counter() - started
    if not obs_list:
        # Gated + well-matched policy: keep a tiny batch so BC does not crash.
        with torch.no_grad():
            teacher_act = teacher.act(env.command)
        obs_all = obs.detach()
        act_all = teacher_act.detach()
        query_count = int(obs_all.shape[0])
        print("  gated: no disagreements this round → kept one fallback batch", flush=True)
    else:
        obs_all = torch.cat(obs_list, dim=0)
        act_all = torch.cat(act_list, dim=0)
    metrics = {
        "queries": float(query_count),
        "visits": float(visit_count),
        "query_rate": float(query_count) / max(visit_count, 1),
        "mean_disagreement": disagree_sum / max(steps, 1),
        "seconds": elapsed,
    }
    print(
        f"  collected {obs_all.shape[0]} labels "
        f"(query_rate={metrics['query_rate']:.3f}, "
        f"disagree={metrics['mean_disagreement']:.4f}) in {elapsed:.1f}s",
        flush=True,
    )
    return obs_all, act_all, metrics


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
    out = args.output or (ns.log_dir / "dagger_trot.pt")

    if args.demo_only != "none":
        which = _resolve_demo_which(args.demo_only, args.save_demos)
        if which is None:
            print("save-demos=none → デモ mp4 は書きません。", flush=True)
            return
        agent = None
        if which in ("all", "bc"):
            load_path = None
            for candidate in (out, ns.log_dir / "dagger_trot.pt", ns.log_dir / "bc_trot.pt"):
                if candidate.is_file():
                    load_path = candidate
                    break
            if load_path is None:
                raise SystemExit(f"DAgger/BC デモ用の重みがありません: {out}")
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
        for path in videos:
            print(f"demo ready: {path.resolve()}", flush=True)
        if videos:
            print(f"demo folder: {videos[0].parent.resolve()}", flush=True)
        return

    init_path = _resolve_init_from(args.init_from, ns.log_dir)
    env = PickAndCarryEnv(cfg=cfg)
    device = env.device
    teacher = TrotTeacher(
        env.num_envs,
        dt=cfg.dt,
        frequency_hz=args.frequency,
        leg_action_scale=cfg.leg_action_scale,
        device=device,
    )

    # Seed dataset: teacher demos and/or empty (when loading BC).
    if init_path is None and args.seed_seconds > 0:
        seed_obs, seed_act = collect_demos(env, teacher, args.seed_seconds)
    else:
        seed_obs = torch.zeros(0, env.num_obs, device=device)
        seed_act = torch.zeros(0, env.num_actions, device=device)
        if init_path is None:
            print("seed-seconds=0 and no --init-from → starting with empty buffer", flush=True)
    env.close()

    env = PickAndCarryEnv(cfg=cfg)
    device = env.device
    if seed_obs.numel() > 0:
        seed_obs = seed_obs.to(device)
        seed_act = seed_act.to(device)
    teacher = TrotTeacher(
        env.num_envs,
        dt=cfg.dt,
        frequency_hz=args.frequency,
        leg_action_scale=cfg.leg_action_scale,
        device=device,
    )

    init_std = initial_action_std(env.num_actions, ns.leg_init_std, ns.arm_init_std)
    agent = ActorCritic(env.num_obs, env.num_actions, init_std=init_std).to(device)
    mask = locomotion_action_mask(env.num_actions, device=device)
    agent.set_action_mask(mask)

    if init_path is not None:
        blob = torch_load(init_path, device)
        if int(blob.get("obs_dim", -1)) != env.num_obs or int(blob.get("act_dim", -1)) != env.num_actions:
            raise SystemExit(
                f"--init-from {init_path} は obs/act "
                f"{blob.get('obs_dim')}/{blob.get('act_dim')} で、"
                f"今の環境は {env.num_obs}/{env.num_actions} です。"
            )
        agent.load_state_dict(blob["model"])
        print(f"student weights from {init_path.resolve()}", flush=True)

    dataset_obs = seed_obs
    dataset_act = seed_act
    tracker = LocalTracker()
    tracker.start(
        ns.log_dir,
        {
            "algorithm": "dagger",
            "dagger_mode": args.mode,
            "curriculum": "walk",
            "arm": cfg.arm,
            "obs_dim": env.num_obs,
            "act_dim": env.num_actions,
            "num_envs": cfg.num_envs,
            "rounds": args.rounds,
            "seconds_per_round": args.seconds,
            "bc_epochs": args.bc_epochs,
            "threshold": args.threshold,
            "beta0": args.beta0,
            "beta_decay": args.beta_decay,
            "init_from": str(init_path) if init_path is not None else None,
            "easy_walk": easy,
        },
    )
    try:
        if dataset_obs.shape[0] > 0 and args.seed_bc_epochs > 0 and init_path is None:
            print(f"seed BC epochs={args.seed_bc_epochs} on {dataset_obs.shape[0]} samples…", flush=True)
            train_bc(
                agent,
                dataset_obs,
                dataset_act,
                epochs=args.seed_bc_epochs,
                batch_size=min(args.batch_size, dataset_obs.shape[0]),
                lr=args.lr,
                action_mask=mask,
                tracker=tracker,
            )

        print(
            f"DAgger mode={args.mode} rounds={args.rounds} "
            f"seconds/round={args.seconds} bc_epochs={args.bc_epochs}",
            flush=True,
        )
        prev_mse: float | None = None
        for round_idx in range(1, args.rounds + 1):
            beta = dagger_beta(round_idx, beta0=args.beta0, decay=args.beta_decay)
            print(f"DAgger round {round_idx}/{args.rounds}  beta={beta:.3f}", flush=True)
            new_obs, new_act, round_metrics = collect_dagger_round(
                env,
                teacher,
                agent,
                seconds=args.seconds,
                beta=beta,
                mode=args.mode,
                threshold=args.threshold,
                action_mask=mask,
            )
            if dataset_obs.shape[0] == 0:
                dataset_obs, dataset_act = new_obs, new_act
            else:
                dataset_obs = torch.cat((dataset_obs, new_obs), dim=0)
                dataset_act = torch.cat((dataset_act, new_act), dim=0)
            print(f"  aggregate size={dataset_obs.shape[0]}", flush=True)
            train_bc(
                agent,
                dataset_obs,
                dataset_act,
                epochs=args.bc_epochs,
                batch_size=min(args.batch_size, dataset_obs.shape[0]),
                lr=args.lr,
                action_mask=mask,
                tracker=None,
                log_every=max(1, args.bc_epochs // 5),
            )
            # One eval loss print for tracker.
            with torch.no_grad():
                pred = agent.actor.mean_action(dataset_obs[: min(4096, dataset_obs.shape[0])])
                tgt = dataset_act[: pred.shape[0]]
                mse = float(((pred - tgt) * mask.expand_as(pred)).pow(2).mean())
            colored = format_metric("agg_mse", mse, prev_mse, higher_is_better=False, fmt=".5f")
            print(
                f"  round {round_idx} done  {colored}  "
                f"query_rate={round_metrics['query_rate']:.3f}",
                flush=True,
            )
            prev_mse = mse
            tracker.log_row(
                {
                    "iteration": round_idx,
                    "beta": beta,
                    "agg_size": float(dataset_obs.shape[0]),
                    "query_rate": round_metrics["query_rate"],
                    "mean_disagreement": round_metrics["mean_disagreement"],
                    "agg_mse": mse,
                }
            )
            _save_dagger_checkpoint(
                out,
                agent=agent,
                obs_dim=env.num_obs,
                act_dim=env.num_actions,
                arm=cfg.arm,
                algo=normalize_algo(ns.algo),
                easy=easy,
                mode=args.mode,
                round_idx=round_idx,
            )
            tracker.log_checkpoint(out)
            print(f"saved DAgger checkpoint {out.resolve()} (round={round_idx})", flush=True)

        env.close()
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
            kind="dagger",
            iteration=int(args.rounds),
            epochs=int(args.bc_epochs),
            extra={"checkpoint": str(out.name), "mode": args.mode},
        )
        videos: list[Path] = []
        if which in ("all", "teacher"):
            videos.extend(
                record_demos(
                    cfg,
                    log_dir=ns.log_dir,
                    seconds=demo_seconds,
                    frequency_hz=args.frequency,
                    agent=None,
                    which="teacher",
                )
            )
        if which in ("all", "bc"):
            cfg._demo_epochs = int(args.bc_epochs)
            policy_videos = record_demos(
                cfg,
                log_dir=ns.log_dir,
                seconds=demo_seconds,
                frequency_hz=args.frequency,
                agent=agent.cpu(),
                which="bc",
                video_dir=video_dir,
            )
            # record_demos writes bc_trot_*; rename for DAgger clarity.
            from pick_and_carry.trot_teacher import DEMO_SKILLS

            renamed: list[Path] = []
            for path in policy_videos:
                name = path.name
                if name.startswith("bc_trot_"):
                    dst = path.with_name("dagger_trot_" + name[len("bc_trot_") :])
                    if path.is_file():
                        path.replace(dst)
                        renamed.append(dst)
                    elif dst.is_file():
                        renamed.append(dst)
                else:
                    renamed.append(path)
            videos.extend(renamed)
        for path in videos:
            print(f"demo ready: {path.resolve()}", flush=True)
        if videos:
            print(f"demo folder: {videos[-1].parent.resolve()}", flush=True)

    print(
        f"next: python -m pick_and_carry.train_ppo --curriculum walk --easy-walk "
        f"--init-from {out} --iterations 1000",
        flush=True,
    )


if __name__ == "__main__":
    main()
