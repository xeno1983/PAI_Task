"""Write the PPO learning curves to an HTML file and print its path.

    .venv\\Scripts\\python scripts\\plot_curves.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pick_and_carry.tracking import load_metrics, load_tensorboard_metrics, render_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Render PPO learning curves.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/ppo"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def default_output() -> Path:
    folder = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"
    if folder.is_dir():
        return folder / "PickAndCarry_curves.html"
    desktop = Path.home() / "Desktop"
    if desktop.is_dir():
        return desktop / "PickAndCarry_curves.html"
    return Path("logs/ppo/curves.html")


def main() -> None:
    args = parse_args()
    output = args.output or default_output()
    metrics_path = args.log_dir / "metrics.jsonl"
    rows = load_metrics(metrics_path)
    source = "metrics.jsonl"
    if not rows:
        rows = load_tensorboard_metrics(args.log_dir)
        source = "tensorboard"
    run = None
    run_path = args.log_dir / "run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
    if not rows:
        print(f"{args.log_dir.resolve()} に学習記録がありません。先にメニューの 1 で学習してください。", flush=True)
        raise SystemExit(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_curves(rows, run), encoding="utf-8")
    print(f"curves: {output.resolve()} ({len(rows)} iterations, {source})", flush=True)


if __name__ == "__main__":
    main()
