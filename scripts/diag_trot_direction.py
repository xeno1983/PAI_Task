"""Verify teacher forward walks toward +X (gripper front).

Locked polarity lives in ``pick_and_carry.cpg``:
  ``_SAGITTAL_SIGN = +1``, ``_CALF_SWING_SIGN = +1`` (thigh/calf same swing sign).

Usage:
  python -u scripts/diag_trot_direction.py
  python -u scripts/diag_trot_direction.py --sweep   # compare sign combos

Exit 0 only when forward Δx > 0.05 m over ~4 s.
"""

from __future__ import annotations

import argparse
import sys

from pick_and_carry import cpg as cpg_mod
from pick_and_carry.config import EnvConfig
from pick_and_carry.trot_teacher import TrotTeacher, skill_command


def _run_forward(env, teacher, cmd, steps: int) -> float:
    env.reset()
    env.set_commands(cmd)
    teacher.reset()
    teacher.set_skill("forward")
    x0 = float(env.base_pos[0, 0].detach().cpu())
    for _ in range(steps):
        env.step(teacher.act(cmd, skill="forward"))
    return float(env.base_pos[0, 0].detach().cpu()) - x0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="Compare sign combinations.")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    from pick_and_carry.env import PickAndCarryEnv

    cfg = EnvConfig(
        num_envs=1,
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
    teacher = TrotTeacher(1, dt=cfg.dt, frequency_hz=2.5, smooth=1.0, device=env.device)
    cmd = skill_command("forward", device=env.device, n=1)
    steps = max(1, int(round(args.seconds / cfg.dt)))

    if args.sweep:
        results = []
        for sag in (+1.0, -1.0):
            for same in (True, False):
                cpg_mod._SAGITTAL_SIGN = sag
                cpg_mod._CALF_SWING_SIGN = 1.0 if same else -1.0
                dx = _run_forward(env, teacher, cmd, steps)
                print(f"sagittal={sag:+.0f} calf_same_sign={same}  dx={dx:+.4f}")
                results.append((dx, sag, same))
        best = max(results, key=lambda row: row[0])
        print(f"BEST dx={best[0]:+.4f} sagittal={best[1]:+.0f} calf_same_sign={best[2]}")
        ok = best[0] > 0.05
    else:
        print(f"_SAGITTAL_SIGN={cpg_mod._SAGITTAL_SIGN} _CALF_SWING_SIGN={cpg_mod._CALF_SWING_SIGN}")
        dx = _run_forward(env, teacher, cmd, steps)
        print(f"forward skill: dx={dx:+.4f} (need > +0.05 toward gripper / +X)")
        ok = dx > 0.05
        # Guard against accidental polarity drift in source constants.
        if cpg_mod._SAGITTAL_SIGN != 1.0 or cpg_mod._CALF_SWING_SIGN != 1.0:
            print("FAIL: expected _SAGITTAL_SIGN=+1 and _CALF_SWING_SIGN=+1")
            ok = False

    print(f"ok_front_is_plus_x={ok}")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
