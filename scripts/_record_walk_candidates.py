"""Record several walk checkpoints for visual comparison."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "logs" / "ppo_curriculum_walk"
OUT = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry" / "walk_candidates"

# Mix of high-tracking, high-reward, and high-gait eras (not only latest).
# Distinct eras: early-stable, high-tracking, high-reward, high-gait.
CANDIDATES = (1020, 1260, 1520, 1900)


def metric_summary(ckpt: int) -> str:
    rows = [
        json.loads(line)
        for line in (LOG / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by = {int(row["iteration"]): row for row in rows}
    window = [by[i] for i in range(ckpt - 5, ckpt + 6) if i in by]
    if not window:
        return ""

    def avg(key: str) -> float:
        vals = [float(row[key]) for row in window if key in row]
        return sum(vals) / len(vals) if vals else 0.0

    return (
        f"track={avg('tracking'):.3f} walk={avg('walk_score'):.3f} "
        f"gait={avg('gait'):.3f} rew={avg('reward'):.3f}"
    )


def record(ckpt: int) -> Path:
    src = LOG / f"ppo_{ckpt:06d}.pt"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"walk_iter{ckpt:06d}.mp4"
    dest = OUT / name
    before = {path.resolve() for path in LOG.rglob(name) if path.is_file()}
    # Also match train_ppo nested naming: PickAndCarry_walk.mp4 under iter folder.
    cmd = [
        str(PYTHON),
        "-u",
        "-m",
        "pick_and_carry.train_ppo",
        "--play",
        "--resume",
        str(src),
        "--log-dir",
        str(LOG),
        "--curriculum",
        "walk",
        "--arm",
        "z1",
        "--num-envs",
        "1",
        "--seconds",
        "10",
        "--record",
        str(OUT / name),
        "--terrain-steps",
        "--control-split",
        "walk",
    ]
    print(f"=== walk {ckpt}: {metric_summary(ckpt)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    # Prefer nested clip created this run.
    nested = sorted(
        LOG.rglob("*.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    chosen = None
    for path in nested:
        if f"iter{ckpt:06d}" in path.as_posix() and path.suffix.lower() == ".mp4":
            chosen = path
            break
    if chosen is None:
        fresh = [path for path in OUT.glob("*.mp4") if path.resolve() not in before]
        chosen = max(fresh or list(OUT.glob(name)), key=lambda path: path.stat().st_mtime)
    shutil.copy2(chosen, dest)
    print(f"saved {dest} from {chosen}", flush=True)
    return dest


def main() -> None:
    paths = [record(ckpt) for ckpt in CANDIDATES]
    print("DONE")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
