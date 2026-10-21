"""Write learning curves to an HTML file and print its path.

    .venv\\Scripts\\python scripts\\plot_curves.py --log-dir logs/ppo_curriculum_walk
    .venv\\Scripts\\python scripts\\plot_curves.py --compare logs/ppo_curriculum_walk logs/grpo_curriculum_walk logs/sym_ppo_curriculum_walk
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

from pick_and_carry.algorithms import ALGO_LABELS, normalize_algo
from pick_and_carry.tracking import load_metrics, load_tensorboard_metrics, render_compare_curves, render_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Render learning curves.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/ppo"))
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        default=None,
        help="Two or more log folders to overlay on one page.",
    )
    parser.add_argument("--labels", nargs="+", default=None, help="Optional labels for --compare.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def default_output(compare: bool = False) -> Path:
    name = "PickAndCarry_compare_curves.html" if compare else "PickAndCarry_curves.html"
    folder = Path(__file__).resolve().parents[1] / "output"
    if folder.is_dir():
        return folder / name
    desktop = Path.home() / "Desktop"
    if desktop.is_dir():
        return desktop / name
    return Path("logs") / name


def load_rows(log_dir: Path) -> tuple[list[dict], dict | None, str]:
    metrics_path = log_dir / "metrics.jsonl"
    rows = load_metrics(metrics_path)
    source = "metrics.jsonl"
    if not rows:
        rows = load_tensorboard_metrics(log_dir)
        source = "tensorboard"
    run = None
    run_path = log_dir / "run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
    return rows, run, source


def label_for(log_dir: Path, run: dict | None, fallback: str | None) -> str:
    if fallback:
        return fallback
    if run and run.get("algorithm"):
        try:
            return ALGO_LABELS[normalize_algo(str(run["algorithm"]))]
        except ValueError:
            return str(run["algorithm"])
    name = log_dir.name
    for key, label in ALGO_LABELS.items():
        if name.startswith(f"{key}_"):
            return label
    return name


def main() -> None:
    args = parse_args()
    if args.compare:
        if len(args.compare) < 2:
            raise SystemExit("--compare には 2 つ以上の保存先を指定してください。")
        labels = list(args.labels or [])
        while len(labels) < len(args.compare):
            labels.append(None)
        runs = []
        for path, label in zip(args.compare, labels):
            rows, run, source = load_rows(path)
            if not rows:
                print(f"記録なし: {path.resolve()}", flush=True)
                continue
            runs.append((label_for(path, run, label), rows, run))
            print(f"compare: {path.resolve()} ({len(rows)} iterations, {source})", flush=True)
        if len(runs) < 2:
            print("比較できる記録が 2 件未満です。先に別の学習手法でも学習してください。", flush=True)
            raise SystemExit(1)
        output = args.output or default_output(compare=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_compare_curves(runs), encoding="utf-8")
        print(f"curves: {output.resolve()}", flush=True)
        return

    output = args.output or default_output(compare=False)
    rows, run, source = load_rows(args.log_dir)
    if not rows:
        print(f"{args.log_dir.resolve()} に学習記録がありません。先にメニューの 1 で学習してください。", flush=True)
        raise SystemExit(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_curves(rows, run), encoding="utf-8")
    print(f"curves: {output.resolve()} ({len(rows)} iterations, {source})", flush=True)


if __name__ == "__main__":
    main()
