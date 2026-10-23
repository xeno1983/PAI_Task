"""Scan metrics for alternative 'looks good' windows for walk/reach/carry."""

from __future__ import annotations

import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ckpts(log_dir: Path) -> list[int]:
    out = []
    for path in log_dir.glob("ppo_*.pt"):
        stem = path.stem
        if stem.startswith("ppo_") and stem[4:].isdigit():
            out.append(int(stem[4:]))
    return sorted(out)


def nearest(iters: list[int], target: int) -> int:
    return min(iters, key=lambda i: abs(i - target))


def rolling(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def report_walk() -> None:
    log = Path("logs/ppo_curriculum_walk")
    data = rows(log / "metrics.jsonl")
    iters = ckpts(log)
    print("=== WALK: look for high tracking AND gait AND walk_score together ===")
    # Also find peaks of tracking alone (user may remember snappier walking)
    for label, key in (("tracking", "tracking"), ("walk_score", "walk_score"), ("gait", "gait"), ("reward", "reward")):
        ranked = sorted(data, key=lambda r: float(r.get(key, 0)), reverse=True)[:6]
        print(f" top {key}:")
        for r in ranked:
            it = int(r["iteration"])
            print(
                f"  iter={it} ckpt~{nearest(iters, it)} "
                f"track={float(r.get('tracking',0)):.3f} "
                f"walk={float(r.get('walk_score',0)):.3f} "
                f"gait={float(r.get('gait',0)):.3f} "
                f"rew={float(r.get('reward',0)):.3f}"
            )
    # periods where all three > thresholds
    hits = []
    for r in data:
        t, w, g = float(r.get("tracking", 0)), float(r.get("walk_score", 0)), float(r.get("gait", 0))
        if t >= 0.45 and w >= 0.47 and g >= 0.42:
            hits.append((t + w + g, int(r["iteration"]), t, w, g, float(r.get("reward", 0))))
    hits.sort(reverse=True)
    print(" balanced high periods (track>=0.45 walk>=0.47 gait>=0.42):")
    seen = set()
    for score, it, t, w, g, rew in hits:
        ck = nearest(iters, it)
        if ck in seen:
            continue
        seen.add(ck)
        print(f"  iter={it} ckpt={ck} track={t:.3f} walk={w:.3f} gait={g:.3f} rew={rew:.3f}")
        if len(seen) >= 8:
            break


def report_reach() -> None:
    log = Path("logs/ppo_curriculum_reach")
    data = rows(log / "metrics.jsonl")
    iters = ckpts(log)
    print("=== REACH: approach is tiny; look reward + success + late stability ===")
    # success moments
    succ = [r for r in data if float(r.get("success", 0) or 0) >= 0.5]
    print(f" success rows: {len(succ)}")
    for r in succ[:15]:
        it = int(r["iteration"])
        print(
            f"  SUCCESS iter={it} ckpt~{nearest(iters, it)} "
            f"approach={float(r.get('approach',0)):.4f} rew={float(r.get('reward',0)):.3f}"
        )
    # high reward windows in mid/late training
    ranked = sorted(
        (r for r in data if int(r["iteration"]) >= 300),
        key=lambda r: float(r.get("reward", 0)),
        reverse=True,
    )[:12]
    print(" top reward iter>=300:")
    seen = set()
    for r in ranked:
        it = int(r["iteration"])
        ck = nearest(iters, it)
        if ck in seen:
            continue
        seen.add(ck)
        print(
            f"  iter={it} ckpt={ck} approach={float(r.get('approach',0)):.4f} "
            f"rew={float(r.get('reward',0)):.3f} grasp={float(r.get('grasp',0)):.3f} "
            f"success={r.get('success')}"
        )


def report_carry() -> None:
    log = Path("logs/ppo_z1")
    data = rows(log / "metrics.jsonl")
    iters = ckpts(log)
    print("=== CARRY: prefer high grasp + low ee_dist + reward (task completion look) ===")
    # High grasp periods (actually holding)
    ranked = sorted(data, key=lambda r: float(r.get("grasp", 0)), reverse=True)[:15]
    print(" top grasp:")
    seen = set()
    for r in ranked:
        it = int(r["iteration"])
        ck = nearest(iters, it)
        if ck in seen:
            continue
        seen.add(ck)
        print(
            f"  iter={it} ckpt={ck} grasp={float(r.get('grasp',0)):.3f} "
            f"ee={float(r.get('ee_dist',0)):.3f} rew={float(r.get('reward',0)):.3f} "
            f"success={r.get('success')}"
        )
    # Composite: grasp high AND ee small AND reward decent, later training
    scored = []
    for r in data:
        it = int(r["iteration"])
        if it < 800:
            continue
        grasp = float(r.get("grasp", 0))
        ee = float(r.get("ee_dist", 1))
        rew = float(r.get("reward", 0))
        if grasp < 0.4:
            continue
        score = 0.5 * grasp + 0.3 * (1.0 / (0.05 + ee)) + 0.2 * rew
        scored.append((score, it, grasp, ee, rew, r.get("success")))
    scored.sort(reverse=True)
    print(" grasp>=0.4 composites:")
    seen = set()
    for score, it, grasp, ee, rew, success in scored:
        ck = nearest(iters, it)
        if ck in seen:
            continue
        seen.add(ck)
        print(f"  score={score:.3f} iter={it} ckpt={ck} grasp={grasp:.3f} ee={ee:.3f} rew={rew:.3f} success={success}")
        if len(seen) >= 10:
            break
    # Also: best ee_dist among grasp>=0.45
    print(" lowest ee among grasp>=0.45:")
    filtered = [
        (float(r.get("ee_dist", 1)), int(r["iteration"]), float(r.get("grasp", 0)), float(r.get("reward", 0)))
        for r in data
        if float(r.get("grasp", 0)) >= 0.45 and int(r["iteration"]) >= 500
    ]
    filtered.sort()
    seen = set()
    for ee, it, grasp, rew in filtered:
        ck = nearest(iters, it)
        if ck in seen:
            continue
        seen.add(ck)
        print(f"  iter={it} ckpt={ck} grasp={grasp:.3f} ee={ee:.3f} rew={rew:.3f}")
        if len(seen) >= 8:
            break


if __name__ == "__main__":
    report_walk()
    report_reach()
    report_carry()
