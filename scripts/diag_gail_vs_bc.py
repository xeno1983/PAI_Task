"""Compare BC vs GAIL checkpoints on a short forward walk (deterministic)."""

from __future__ import annotations

from pathlib import Path

import torch

from pick_and_carry.agents import ActorCritic
from pick_and_carry.checkpoints import torch_load
from pick_and_carry.config import EnvConfig
from pick_and_carry.curriculum import locomotion_action_mask
from pick_and_carry.env import PickAndCarryEnv


def _rms_drift(base: dict, other: dict) -> float:
    vals = []
    for key, tensor in base.items():
        if key in other and torch.is_floating_point(tensor):
            vals.append(float((other[key].float() - tensor.float()).pow(2).mean().sqrt()))
    return sum(vals) / max(len(vals), 1)


def eval_forward(env: PickAndCarryEnv, agent: ActorCritic, *, seconds: float = 3.0) -> dict[str, float]:
    obs = env.reset()
    cmd = torch.zeros(env.num_envs, 3, device=env.device)
    cmd[:, 0] = 0.35
    env.set_commands(cmd)
    env.hold_commands = True
    x0 = env.base_pos[:, 0].clone()
    steps = max(1, int(round(seconds / env.cfg.dt)))
    amp_vals: list[float] = []
    falls = 0
    for _ in range(steps):
        with torch.no_grad():
            action = agent.act_mean(obs)
            obs, _rew, done, _info = env.step(action)
        amp_vals.append(float(action[:, 1].abs().mean()))
        falls += int(done.sum().item())
    dx = float((env.base_pos[:, 0] - x0).mean())
    return {
        "dx": dx,
        "amp_abs": sum(amp_vals) / len(amp_vals),
        "falls": float(falls),
    }


def main() -> None:
    root = Path("logs/ppo_curriculum_walk_easy")
    cfg = EnvConfig(
        num_envs=4,
        backend="gpu",
        show_viewer=False,
        curriculum="walk",
        control_split="walk",
        terrain=False,
        terrain_steps=False,
        arm_disturb=False,
        push_robots=False,
        randomize_friction=False,
    )
    env = PickAndCarryEnv(cfg=cfg)
    device = env.device
    bc_blob = torch_load(root / "bc_trot.pt", torch.device("cpu"))
    bc_model = bc_blob["model"]

    names = [
        "bc_trot.pt",
        "gail_000050.pt",
        "gail_000200.pt",
        "gail_000500.pt",
        "gail_001000.pt",
    ]
    print(f"{'ckpt':22s}  {'dx_3s':>8s}  {'amp':>7s}  {'falls':>6s}  {'drift':>8s}  gail_iter")
    for name in names:
        path = root / name
        if not path.is_file():
            print(f"{name:22s}  MISSING")
            continue
        blob = torch_load(path, device)
        agent = ActorCritic(int(blob["obs_dim"]), int(blob["act_dim"]), init_std=0.3).to(device)
        agent.load_state_dict(blob["model"])
        agent.set_action_mask(locomotion_action_mask(int(blob["act_dim"]), device=device))
        agent.eval()
        stats = eval_forward(env, agent, seconds=3.0)
        drift = _rms_drift(bc_model, {k: v.detach().cpu() for k, v in blob["model"].items()})
        print(
            f"{name:22s}  {stats['dx']:+8.3f}  {stats['amp_abs']:7.3f}  "
            f"{int(stats['falls']):6d}  {drift:8.4f}  {blob.get('gail_iteration')}"
        )
    env.close()


if __name__ == "__main__":
    main()
