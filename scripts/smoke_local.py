"""One-process check that Genesis can load the robot and step.

Run from the repository root with the project virtualenv:

    .venv\\Scripts\\python scripts\\smoke_local.py
    .venv\\Scripts\\python scripts\\smoke_local.py --viewer --seconds 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Genesis prints box-drawing characters. The Windows cp932 console cannot encode them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

from pick_and_carry.config import EnvConfig
from pick_and_carry.env import PickAndCarryEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the pick-and-carry scene and step a zero action.")
    parser.add_argument("--viewer", action="store_true", help="Open the Genesis 3D window.")
    parser.add_argument("--record", type=str, default=None, help="Save an mp4 of the run to this path.")
    parser.add_argument("--seconds", type=float, default=None, help="How long to hold the standing pose.")
    parser.add_argument("--steps", type=int, default=None, help="Step count. Overrides --seconds.")
    args = parser.parse_args()
    if args.steps is not None:
        steps = args.steps
    elif args.seconds is not None:
        steps = max(1, int(round(args.seconds / 0.02)))
    else:
        steps = 400 if args.viewer else 5

    print("python", sys.version.split()[0], flush=True)
    print("torch", torch.__version__, "cuda_build", torch.version.cuda, flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Install the cu128 PyTorch wheel before Genesis.")
    print("gpu", torch.cuda.get_device_name(0), flush=True)
    if args.viewer:
        print("Opening the 3D window. The first launch can take one to three minutes.", flush=True)
    else:
        print("Building the scene without a 3D window.", flush=True)

    started = time.perf_counter()
    env = PickAndCarryEnv(
        cfg=EnvConfig(num_envs=1, backend="gpu", show_viewer=args.viewer, record_path=args.record, seed=0)
    )
    print(f"scene ready in {time.perf_counter() - started:.1f}s", flush=True)
    print("urdf", env.urdf_path, flush=True)
    print("ee", env.ee_link_name, "obs", env.num_obs, "act", env.num_actions, flush=True)

    obs = env.reset()
    print("reset", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()), flush=True)
    action = torch.zeros((env.num_envs, env.num_actions), device=obs.device)
    report_every = max(1, steps // 5)
    for step in range(steps):
        obs, reward, done, info = env.step(action)
        if step % report_every == 0 or step == steps - 1:
            print(
                f"step {step} reward {float(reward[0]):.4f} "
                f"base_z {float(env.base_pos[0, 2]):.3f} "
                f"obj {env.obj_pos[0].detach().cpu().tolist()} "
                f"done {bool(done[0])}",
                flush=True,
            )
    env.close()
    print("smoke ok", flush=True)
    if args.record:
        print("video", args.record, flush=True)


if __name__ == "__main__":
    main()
