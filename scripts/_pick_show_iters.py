"""Pick visually strong checkpoints for walk / reach / carry show reels."""

from __future__ import annotations

import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def available_iters(log_dir: Path, prefixes: tuple[str, ...] = ("ppo_",)) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for prefix in prefixes:
        for path in log_dir.glob(f"{prefix}*.pt"):
            stem = path.stem
            if stem.startswith(prefix) and stem[len(prefix) :].isdigit():
                found.append((int(stem[len(prefix) :]), path))
    return sorted(found, key=lambda item: item[0])


def nearest(iters: list[tuple[int, Path]], target: int) -> tuple[int, Path] | None:
    if not iters:
        return None
    return min(iters, key=lambda item: abs(item[0] - target))


def rank(
    name: str,
    log_dir: Path,
    score_fn,
    *,
    prefixes: tuple[str, ...] = ("ppo_",),
    min_iter: int = 0,
    top: int = 8,
) -> list[tuple[float, int, int, Path, dict]]:
    metrics = log_dir / "metrics.jsonl"
    rows = load_rows(metrics)
    iters = available_iters(log_dir, prefixes)
    print(f"=== {name}: {len(iters)} checkpoints, {len(rows)} metric rows")
    scored: list[tuple[float, int, dict]] = []
    for row in rows:
        iteration = int(row["iteration"])
        if iteration < min_iter:
            continue
        try:
            score = float(score_fn(row))
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            continue
        scored.append((score, iteration, row))
    scored.sort(reverse=True, key=lambda item: item[0])
    picks: list[tuple[float, int, int, Path, dict]] = []
    seen: set[int] = set()
    for score, iteration, row in scored:
        hit = nearest(iters, iteration)
        if hit is None or hit[0] in seen:
            continue
        seen.add(hit[0])
        picks.append((score, iteration, hit[0], hit[1], row))
        if len(picks) >= top:
            break
    for score, metric_iter, ckpt_iter, path, row in picks:
        bits = []
        for key in ("tracking", "walk_score", "gait", "approach", "reward", "grasp", "ee_dist", "success"):
            if key in row and isinstance(row[key], (int, float)):
                bits.append(f"{key}={float(row[key]):.3f}")
        print(f"  score={score:.3f} metric={metric_iter} ckpt={ckpt_iter} ({path.name})  " + " ".join(bits))
    return picks


def main() -> None:
    walk = rank(
        "walk ppo",
        Path("logs/ppo_curriculum_walk"),
        lambda r: 0.5 * float(r["tracking"]) + 0.3 * float(r.get("walk_score", 0)) + 0.2 * float(r.get("gait", 0)),
        min_iter=800,
    )
    rank(
        "walk grpo",
        Path("logs/grpo_curriculum_walk"),
        lambda r: 0.5 * float(r["tracking"]) + 0.3 * float(r.get("walk_score", 0)) + 0.2 * float(r.get("gait", 0)),
        prefixes=("grpo_", "ppo_"),
        min_iter=400,
    )
    reach = rank(
        "reach",
        Path("logs/ppo_curriculum_reach"),
        lambda r: 0.55 * float(r.get("approach", 0)) + 0.45 * float(r.get("reward", 0)),
        min_iter=400,
    )
    carry = rank(
        "carry",
        Path("logs/ppo_z1"),
        lambda r: (
            0.40 * float(r.get("grasp", 0))
            + 0.40 * (1.0 / (0.05 + float(r.get("ee_dist", 1.0))))
            + 0.20 * float(r.get("reward", 0))
        ),
        min_iter=2500,
    )
    chosen = {
        "walk": walk[0] if walk else None,
        "reach": reach[0] if reach else None,
        "carry": carry[0] if carry else None,
    }
    print("CHOSEN")
    for stage, item in chosen.items():
        if item is None:
            print(stage, "NONE")
            continue
        _score, _metric, ckpt_iter, path, _row = item
        print(stage, ckpt_iter, path)


if __name__ == "__main__":
    main()
