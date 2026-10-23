"""Record submission show reels from visually strong checkpoints.

Chooses iterations by metrics (not merely latest), then writes preferred
basenames into the launcher folder so ``make_presentation`` picks them up.

    .venv\\Scripts\\python scripts\\record_show_reels.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LAUNCHER = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"

# Prefer periods that look task-complete (grasp/success/gait), not only late ee_dist.
# walk 1620: high tracking + walk_score + best reward band
# reach 760: success=1 with strong reward
# carry 1240: high grasp + success (mid training holds the box; late 5k often only approaches)
SHOWS = (
    {
        "label": "歩行",
        "log_dir": ROOT / "logs" / "ppo_curriculum_walk",
        "ckpt": "ppo_001620.pt",
        "out_name": "PickAndCarry_walk.mp4",
        "curriculum": "walk",
        "seconds": 14.0,
        "extra": ["--terrain-steps", "--control-split", "walk"],
    },
    {
        "label": "到達",
        "log_dir": ROOT / "logs" / "ppo_curriculum_reach",
        "ckpt": "ppo_000760.pt",
        "out_name": "PickAndCarry_reach.mp4",
        "curriculum": "reach",
        "seconds": 14.0,
        "extra": ["--control-split", "both"],
    },
    {
        "label": "箱運び",
        "log_dir": ROOT / "logs" / "ppo_z1",
        "ckpt": "ppo_001240.pt",
        "out_name": "PickAndCarry_z1.mp4",
        "curriculum": "carry",
        "seconds": 20.0,
        "extra": ["--terrain-steps", "--control-split", "both"],
    },
)


def record(show: dict) -> Path:
    log_dir: Path = show["log_dir"]
    ckpt = log_dir / show["ckpt"]
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    LAUNCHER.mkdir(parents=True, exist_ok=True)
    out = LAUNCHER / show["out_name"]
    cmd = [
        str(PYTHON),
        "-u",
        "-m",
        "pick_and_carry.train_ppo",
        "--play",
        "--resume",
        str(ckpt),
        "--log-dir",
        str(log_dir),
        "--curriculum",
        show["curriculum"],
        "--arm",
        "z1",
        "--num-envs",
        "1",
        "--seconds",
        str(show["seconds"]),
        "--record",
        str(out),
        *show["extra"],
    ]
    print(f"=== {show['label']}: {ckpt.name} → {out.name}", flush=True)
    print("+", " ".join(cmd), flush=True)
    before = {path.resolve() for path in log_dir.rglob(show["out_name"]) if path.is_file()}
    subprocess.run(cmd, cwd=ROOT, check=True)
    # train_ppo nests under videos/; always promote the newest nested file for this basename.
    matches = sorted(
        (path for path in log_dir.rglob(show["out_name"]) if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise SystemExit(f"recording not found for {show['label']}")
    # Prefer a file created by this run when possible.
    fresh = [path for path in matches if path.resolve() not in before] or matches
    chosen = fresh[0]
    shutil.copy2(chosen, out)
    print(f"copied nested reel → {out} ({chosen})", flush=True)
    meta = {
        "label": show["label"],
        "checkpoint": str(ckpt),
        "iteration": int(show["ckpt"].split("_")[1].split(".")[0]),
        "output": str(out),
        "seconds": show["seconds"],
    }
    (LAUNCHER / f"{out.stem}_show.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {out} ({out.stat().st_size} bytes)", flush=True)
    return out


def main() -> None:
    if not PYTHON.is_file():
        raise SystemExit(f"python missing: {PYTHON}")
    outputs = [record(show) for show in SHOWS]
    print("DONE")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
