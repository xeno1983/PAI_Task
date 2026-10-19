"""Behavioral cloning of a diagonal-trot teacher, then save a PPO-compatible checkpoint.

Typical use (writes under ``logs/<algo>_curriculum_walk_easy/imitation/``):

    python -m pick_and_carry.train_bc --easy-walk --seconds 90 --epochs 3000

Demos only (teacher / BC policy):

    python -m pick_and_carry.train_bc --demo-only teacher
    python -m pick_and_carry.train_bc --demo-only bc
    python -m pick_and_carry.train_bc --demo-only all

Teacher demos write one mp4 per skill under
``--log-dir/videos/<task>/<kind>_ep…_iter…_<stamp>/``:
stand / sit / jump / forward / backward / turn_left / turn_right / stop.

Then warm-start PPO:

    python -m pick_and_carry.train_ppo --curriculum walk --easy-walk --init-from bc --iterations 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pick_and_carry.agents import ActorCritic
from pick_and_carry.algorithms import normalize_algo
from pick_and_carry.checkpoints import save_checkpoint, torch_load
from pick_and_carry.config import ACTION_LAYOUT, OBS_LAYOUT, layout_slices
from pick_and_carry.cpg import command_is_moving
from pick_and_carry.curriculum import (
    imitation_log_dir,
    locomotion_action_mask,
    nest_imitation_log_dir,
)
from pick_and_carry.console import format_metric
from pick_and_carry.mlflow_log import LocalTracker
from pick_and_carry.train_ppo import initial_action_std, make_config
from pick_and_carry.trot_teacher import (
    DEMO_SKILL_SECONDS,
    DEMO_SKILLS,
    TrotTeacher,
    skill_command,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect trot demos and train behavioral cloning.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=60.0, help="Simulated seconds of teacher rollouts.")
    parser.add_argument("--epochs", type=int, default=3000)
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
    parser.add_argument("--frequency", type=float, default=2.5, help="Trot frequency in Hz.")
    parser.add_argument(
        "--demo-seconds",
        type=float,
        default=0.0,
        help="If >0, override per-skill demo length (every skill uses this many seconds). "
        "0 keeps the built-in lengths (stand/sit/jump/walk/turn/stop).",
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
    parser.add_argument(
        "--save-demos",
        choices=("all", "teacher", "bc", "none"),
        default="all",
        help="Which demo mp4s to write after BC training. "
        "With --demo-only, this still filters (e.g. teacher-only runs ignore 'bc').",
    )
    parser.add_argument("--record", type=str, default=None, help="Ignored; demos use fixed names under --log-dir.")
    return parser.parse_args()


def _resolve_demo_which(demo_only: str, save_demos: str) -> str | None:
    """Return ``record_demos`` which, or None to skip writing mp4s."""
    if demo_only != "none":
        if save_demos == "none":
            return None
        if demo_only == "teacher":
            return "teacher" if save_demos in ("all", "teacher") else None
        if demo_only == "bc":
            return "bc" if save_demos in ("all", "bc") else None
        return save_demos  # demo_only=all → all/teacher/bc
    if save_demos == "none":
        return None
    return save_demos


def _build_walk_args(bc_args):
    """Reuse train_ppo config builder with a fake argparse namespace."""
    easy = bool(bc_args.easy_walk) and not bool(bc_args.no_easy_walk)
    algo = normalize_algo(bc_args.algo)
    log_dir = bc_args.log_dir
    if log_dir is None:
        log_dir = imitation_log_dir(algo=algo, easy=easy)
    else:
        # Launcher may pass the RL walk folder; nest under imitation/.
        log_dir = nest_imitation_log_dir(log_dir)
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
    # Full skill reel is ~43 s; default episode 20 s would timeout mid-clip and
    # (before hold-aware reset) inject a walk command after stand/sit/jump.
    cfg.episode_length_s = max(float(cfg.episode_length_s), 120.0)
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.record_path = str(record_path.resolve())
    return cfg


def _skill_seconds(override: float) -> dict[str, float]:
    if override and override > 0:
        return {name: float(override) for name in DEMO_SKILLS}
    return dict(DEMO_SKILL_SECONDS)


def collect_demos(env, teacher: TrotTeacher, seconds: float) -> tuple[torch.Tensor, torch.Tensor]:
    steps = max(1, int(round(seconds / env.cfg.dt)))
    resample_every = max(1, int(round(2.5 / env.cfg.dt)))
    obs_list: list[torch.Tensor] = []
    act_list: list[torch.Tensor] = []
    obs = env.reset()
    teacher.reset()
    teacher.sample_skills()
    # Lock teacher commands into both env.command and the obs command slice.
    obs = env.set_commands(teacher.commands_for_skills(), hold=True)
    print(f"collecting {steps} steps × {env.num_envs} envs (~{seconds:.0f}s, mixed skills)…", flush=True)
    started = time.perf_counter()
    for step in range(steps):
        if step > 0 and step % resample_every == 0:
            teacher.sample_skills()
            obs = env.set_commands(teacher.commands_for_skills(), hold=True)
        action = teacher.act(env.command)
        obs_list.append(obs.detach())
        act_list.append(action.detach())
        obs, _, done, _ = env.step(action)
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            teacher.reset(done_ids)
            teacher.sample_skills()
            # Re-sample all; simpler than per-env partial update for BC.
            obs = env.set_commands(teacher.commands_for_skills(), hold=True)
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"  collect {step + 1}/{steps}", flush=True)
    env.hold_commands = False
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
    still_amp_weight: float = 8.0,
    tracker: LocalTracker | None = None,
    log_every: int | None = None,
) -> None:
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    n = obs.shape[0]
    mask = action_mask.to(dtype=obs.dtype, device=obs.device)
    cmd_slice = layout_slices(OBS_LAYOUT)["command"]
    amp_index = layout_slices(ACTION_LAYOUT)["cpg"].start + 1
    # Dense enough for MLflow curves without flooding 3000-epoch runs.
    every = int(log_every) if log_every and log_every > 0 else max(1, epochs // 100)
    print(
        f"BC epochs={epochs} batch={batch_size} samples={n} "
        f"still_amp_weight={still_amp_weight}",
        flush=True,
    )
    prev_mse: float | None = None
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, device=obs.device)
        total = 0.0
        count = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch_obs = obs[idx]
            pred = agent.actor.mean_action(batch_obs)
            err = (pred - actions[idx]).pow(2)
            # stand/sit/jump/stop share command≈0; heavily penalize amp leakage there.
            still = ~command_is_moving(batch_obs[:, cmd_slice])
            amp_w = torch.where(
                still,
                torch.full_like(err[:, amp_index], float(still_amp_weight)),
                torch.ones_like(err[:, amp_index]),
            )
            err = err.clone()
            err[:, amp_index] = err[:, amp_index] * amp_w
            loss = (err * mask.expand_as(err)).sum(dim=-1).mean() / mask.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        leg_mse = total / max(count, 1)
        if epoch == 1 or epoch % every == 0 or epoch == epochs:
            if tracker is not None:
                tracker.log_row({"iteration": epoch, "bc_leg_mse": leg_mse})
            if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                colored = format_metric(
                    "leg_mse",
                    leg_mse,
                    prev_mse,
                    higher_is_better=False,
                    fmt=".5f",
                )
                print(f"  bc epoch {epoch}/{epochs}  {colored}", flush=True)
                prev_mse = leg_mse


def _record_skill_clip(
    *,
    cfg,
    log_dir: Path,
    skill: str,
    seconds: float,
    frequency_hz: float,
    path: Path,
) -> Path:
    from pick_and_carry.env import PickAndCarryEnv

    demo_cfg = _demo_cfg(cfg, log_dir, path)
    env = PickAndCarryEnv(cfg=demo_cfg)
    teacher = TrotTeacher(
        1,
        dt=demo_cfg.dt,
        frequency_hz=frequency_hz,
        leg_action_scale=demo_cfg.leg_action_scale,
        device=env.device,
    )
    env.reset()
    teacher.reset()
    teacher.set_skill(skill)
    cmd = skill_command(skill, device=env.device, n=1)
    env.set_commands(cmd, hold=True)
    steps = max(1, int(round(seconds / demo_cfg.dt)))
    print(f"  {skill:12s} {seconds:.1f}s → {path.name}", flush=True)
    for _ in range(steps):
        action = teacher.act(env.command, skill=skill)
        _, _, done, _ = env.step(action)
        if bool(done.any()):
            teacher.reset()
            teacher.set_skill(skill)
            env.set_commands(cmd, hold=True)
    env.close()
    return path


def _record_bc_skill_clip(
    *,
    cfg,
    log_dir: Path,
    skill: str,
    seconds: float,
    agent: ActorCritic,
    path: Path,
) -> Path:
    from pick_and_carry.env import PickAndCarryEnv

    demo_cfg = _demo_cfg(cfg, log_dir, path)
    env = PickAndCarryEnv(cfg=demo_cfg)
    agent = agent.to(env.device)
    obs = env.reset()
    cmd = skill_command(skill, device=env.device, n=1)
    obs = env.set_commands(cmd, hold=True)
    steps = max(1, int(round(seconds / demo_cfg.dt)))
    print(f"  {skill:12s} {seconds:.1f}s → {path.name}", flush=True)
    with torch.no_grad():
        for _ in range(steps):
            action = agent.act_mean(obs)
            obs, _, done, _ = env.step(action)
            if bool(done.any()):
                # Re-lock still/walk command after auto-reset (fall/timeout).
                obs = env.set_commands(cmd, hold=True)
    env.close()
    return path


def _record_teacher_reel(
    *,
    cfg,
    log_dir: Path,
    lengths: dict[str, float],
    frequency_hz: float,
    path: Path,
) -> Path:
    """One mp4 with stand→sit→jump→walk→turn→stop in order."""
    from pick_and_carry.env import PickAndCarryEnv

    demo_cfg = _demo_cfg(cfg, log_dir, path)
    env = PickAndCarryEnv(cfg=demo_cfg)
    teacher = TrotTeacher(
        1,
        dt=demo_cfg.dt,
        frequency_hz=frequency_hz,
        leg_action_scale=demo_cfg.leg_action_scale,
        device=env.device,
    )
    env.reset()
    teacher.reset()
    total = sum(lengths[s] for s in DEMO_SKILLS)
    print(f"  reel         {total:.0f}s (all skills) → {path.name}", flush=True)
    for skill in DEMO_SKILLS:
        teacher.set_skill(skill)
        cmd = skill_command(skill, device=env.device, n=1)
        env.set_commands(cmd, hold=True)
        steps = max(1, int(round(lengths[skill] / demo_cfg.dt)))
        print(f"    … {skill} ({lengths[skill]:.1f}s)", flush=True)
        for _ in range(steps):
            action = teacher.act(env.command, skill=skill)
            _, _, done, _ = env.step(action)
            if bool(done.any()):
                teacher.reset()
                teacher.set_skill(skill)
                env.set_commands(cmd, hold=True)
    env.close()
    return path


def _record_bc_reel(
    *,
    cfg,
    log_dir: Path,
    lengths: dict[str, float],
    agent: ActorCritic,
    path: Path,
) -> Path:
    from pick_and_carry.env import PickAndCarryEnv

    demo_cfg = _demo_cfg(cfg, log_dir, path)
    env = PickAndCarryEnv(cfg=demo_cfg)
    agent = agent.to(env.device)
    obs = env.reset()
    total = sum(lengths[s] for s in DEMO_SKILLS)
    print(f"  reel         {total:.0f}s (all skills) → {path.name}", flush=True)
    with torch.no_grad():
        for skill in DEMO_SKILLS:
            cmd = skill_command(skill, device=env.device, n=1)
            obs = env.set_commands(cmd, hold=True)
            steps = max(1, int(round(lengths[skill] / demo_cfg.dt)))
            print(f"    … {skill} ({lengths[skill]:.1f}s)", flush=True)
            for _ in range(steps):
                action = agent.act_mean(obs)
                obs, _, done, _ = env.step(action)
                if bool(done.any()):
                    obs = env.set_commands(cmd, hold=True)
    env.close()
    return path


def record_demos(
    cfg,
    *,
    log_dir: Path,
    seconds: float,
    frequency_hz: float,
    agent: ActorCritic | None,
    which: str = "all",
    video_dir: Path | None = None,
) -> list[Path]:
    """Record a skill reel plus one mp4 per skill under ``video_dir`` (or ``log_dir``)."""
    from pick_and_carry.video_paths import make_video_run_dir, video_task_name

    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    lengths = _skill_seconds(seconds)
    paths: list[Path] = []
    easy = bool(getattr(cfg, "easy_walk", False) or "walk_easy" in log_dir.name)
    task = video_task_name(getattr(cfg, "curriculum", "walk"), easy=easy)
    print(f"demo videos root: {log_dir / 'videos'}  task={task}", flush=True)

    def _out(kind: str, epochs: int | None = None) -> Path:
        if video_dir is not None:
            out = Path(video_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            return out
        out = make_video_run_dir(
            log_dir,
            task=task,
            kind=kind,
            epochs=epochs,
            extra={"curriculum": getattr(cfg, "curriculum", "walk"), "which": which},
        )
        print(f"  run folder: {out}", flush=True)
        return out

    if which in ("all", "teacher"):
        out_dir = _out("teacher")
        print(f"recording teacher demos → {out_dir}", flush=True)
        reel = out_dir / "trot_teacher_demo.mp4"
        paths.append(
            _record_teacher_reel(
                cfg=cfg,
                log_dir=log_dir,
                lengths=lengths,
                frequency_hz=frequency_hz,
                path=reel,
            )
        )
        print(f"video: {reel.resolve()}", flush=True)
        for skill in DEMO_SKILLS:
            path = out_dir / f"trot_teacher_{skill}.mp4"
            paths.append(
                _record_skill_clip(
                    cfg=cfg,
                    log_dir=log_dir,
                    skill=skill,
                    seconds=lengths[skill],
                    frequency_hz=frequency_hz,
                    path=path,
                )
            )
            print(f"video: {path.resolve()}", flush=True)

    if which in ("all", "bc"):
        if agent is None:
            raise SystemExit("BC デモには学習済みの agent か bc_trot.pt が必要です。")
        epochs = int(getattr(cfg, "_demo_epochs", 0) or 0) or None
        out_dir = _out("bc", epochs=epochs)
        print(f"recording BC demos → {out_dir}", flush=True)
        reel = out_dir / "bc_trot_demo.mp4"
        paths.append(
            _record_bc_reel(
                cfg=cfg,
                log_dir=log_dir,
                lengths=lengths,
                agent=agent,
                path=reel,
            )
        )
        print(f"video: {reel.resolve()}", flush=True)
        for skill in DEMO_SKILLS:
            path = out_dir / f"bc_trot_{skill}.mp4"
            paths.append(
                _record_bc_skill_clip(
                    cfg=cfg,
                    log_dir=log_dir,
                    skill=skill,
                    seconds=lengths[skill],
                    agent=agent,
                    path=path,
                )
            )
            print(f"video: {path.resolve()}", flush=True)
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
        which = _resolve_demo_which(args.demo_only, args.save_demos)
        if which is None:
            print("save-demos=none → デモ mp4 は書きません。", flush=True)
            return
        agent = None
        if which in ("all", "bc"):
            if not out.is_file():
                raise SystemExit(f"BC デモ用の重みがありません: {out}")
            blob = torch_load(out, torch.device("cpu"))
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

    tracker = LocalTracker()
    tracker.start(
        ns.log_dir,
        {
            "algorithm": "bc",
            "curriculum": "walk",
            "arm": cfg.arm,
            "obs_dim": int(obs.shape[-1]),
            "act_dim": int(actions.shape[-1]),
            "num_envs": cfg.num_envs,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "seed": args.seed,
            "easy_walk": easy,
            "demo_seconds": args.seconds,
        },
    )
    try:
        train_bc(
            agent,
            obs.to(device),
            actions.to(device),
            epochs=args.epochs,
            batch_size=min(args.batch_size, obs.shape[0]),
            lr=args.lr,
            action_mask=mask,
            tracker=tracker,
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
        tracker.log_checkpoint(out)
        print(f"saved BC checkpoint {out.resolve()}", flush=True)
    finally:
        tracker.close()

    which = _resolve_demo_which("none", args.save_demos)
    if which is None:
        print("save-demos=none → デモ mp4 はスキップします。", flush=True)
    else:
        cfg._demo_epochs = int(args.epochs)
        videos = record_demos(
            cfg,
            log_dir=ns.log_dir,
            seconds=demo_seconds,
            frequency_hz=args.frequency,
            agent=agent.cpu(),
            which=which,
        )
        for path in videos:
            print(f"demo ready: {path.resolve()}", flush=True)
        if videos:
            print(f"demo folder: {videos[0].parent.resolve()}", flush=True)
    print(
        f"next: python -m pick_and_carry.train_ppo --curriculum walk --easy-walk "
        f"--init-from {out} --iterations 1000",
        flush=True,
    )


if __name__ == "__main__":
    main()
