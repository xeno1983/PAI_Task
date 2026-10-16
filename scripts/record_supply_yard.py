"""Record a wide yard where a scripted trot picks up and carries several boxes.

The walking gait is the fixed trot, and the arm is a fixed reach pose. This is a
scene recording, not the policy that is training on the single-box course.

    .venv\\Scripts\\python scripts\\record_supply_yard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pick_and_carry.config import EnvConfig
from pick_and_carry.env import PickAndCarryEnv

# Low reach, as joint offsets from the default arm, divided by arm_action_scale 1.2.
REACH = (0.0, 0.583, 0.0, 0.333, 0.0, 0.333, 0.0)


def follow(env: PickAndCarryEnv) -> None:
    base = env.base_pos[0]
    x = float(base[0])
    y = float(base[1])
    env.camera.set_pose(pos=(x - 1.2, y - 7.4, 4.0), lookat=(x + 1.6, y, 0.05))


def action_for(env: PickAndCarryEnv) -> torch.Tensor:
    action = torch.zeros((1, env.num_actions), device=env.device)
    if bool(env.supply_finished[0]):
        return action
    phase = int(env.phase[0])
    if phase == 1:
        action[0, 12:19] = torch.tensor(REACH, device=env.device)
        action[0, 19] = -1.0
    elif phase == 3:
        action[0, 12:19] = torch.tensor(REACH, device=env.device)
        # Stay closed until the hand is low, so the box rides down onto the pad.
        action[0, 19] = 1.0 if float(env.ee_pos[0, 2]) < 0.16 else -1.0
    return action


def main() -> None:
    out = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry" / "PickAndCarry_supplies_90s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(
        num_envs=1,
        backend="gpu",
        seed=2,
        yard_stations=6,
        randomize_terrain=False,
        cycle_supplies=True,
        scripted_gait=True,
        assist_grasp=True,
        terrain_amplitude=0.05,
        episode_length_s=120.0,
        pickup_distance=0.40,
        release_distance=0.22,
        goal_radius=0.60,
        place_height=0.20,
        base_init_pos=(0.15, 0.0, 0.42),
        camera_res=(1280, 720),
        camera_fov=52.0,
        record_path=str(out),
    )
    print("building the wide supply yard", flush=True)
    env = PickAndCarryEnv(cfg=cfg)
    follow(env)
    steps = int(round(90.0 / cfg.dt))
    delivered = 0
    obs = env.reset()
    del obs
    print(f"recording {steps} steps to {out}", flush=True)
    for step in range(steps):
        follow(env)
        before = int(env._supply_index[0])
        finished = bool(env.supply_finished[0])
        env.step(action_for(env))
        now = int(env._supply_index[0])
        if now > before or (bool(env.supply_finished[0]) and not finished):
            delivered += 1
            print(f"t={step * cfg.dt:.1f}s delivered {delivered}", flush=True)
        if step % int(round(5.0 / cfg.dt)) == 0:
            base = env.base_pos[0]
            _, _, yaw_err = env._aim_error(env.goal_pos if int(env.phase[0]) >= 2 else env.obj_pos)
            print(
                f"t={step * cfg.dt:.0f}s base=({float(base[0]):.2f},{float(base[1]):.2f}) "
                f"phase={int(env.phase[0])} supply={now} yaw_err={float(yaw_err[0]):.2f}",
                flush=True,
            )
    env.close()
    print(f"video: {out}", flush=True)


if __name__ == "__main__":
    main()
