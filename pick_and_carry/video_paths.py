"""Video output layout: ``{log_dir}/videos/{task}/{run_id}/*.mp4``.

``task`` names the curriculum (e.g. ``walk_easy``, ``carry``, ``catch_slow``).
``run_id`` encodes which model produced the clip (teacher / bc / gail / ppo) and
how far training had gone (epochs or PPO iteration), plus a timestamp so repeated
runs do not overwrite each other.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def video_task_name(
    curriculum: str,
    *,
    easy: bool = False,
    catch_level: str = "static",
) -> str:
    stage = str(curriculum or "walk")
    if stage == "walk" and easy:
        return "walk_easy"
    if stage == "catch":
        return f"catch_{catch_level or 'static'}"
    if stage == "carry":
        return "carry"
    return stage


def _safe_token(value: str) -> str:
    cleaned = _SAFE.sub("_", str(value).strip())
    return cleaned.strip("._") or "run"


def make_run_id(
    kind: str,
    *,
    iteration: int | None = None,
    epochs: int | None = None,
    stamp: str | None = None,
) -> str:
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [_safe_token(kind)]
    if epochs is not None:
        parts.append(f"ep{int(epochs):04d}")
    if iteration is not None:
        parts.append(f"iter{int(iteration):06d}")
    parts.append(_safe_token(stamp))
    return "_".join(parts)


def videos_root(log_dir: Path | str) -> Path:
    return Path(log_dir) / "videos"


def make_video_run_dir(
    log_dir: Path | str,
    *,
    task: str,
    kind: str,
    iteration: int | None = None,
    epochs: int | None = None,
    stamp: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Create ``videos/{task}/{run_id}/`` and write a small ``run.json`` manifest."""
    run_id = make_run_id(kind, iteration=iteration, epochs=epochs, stamp=stamp)
    path = (videos_root(Path(log_dir).resolve()) / _safe_token(task) / run_id).resolve()
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": task,
        "kind": kind,
        "run_id": run_id,
        "iteration": iteration,
        "epochs": epochs,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        manifest.update(extra)
    (path / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def latest_video_run(
    log_dir: Path | str,
    *,
    task: str | None = None,
    kind: str | None = None,
) -> Path | None:
    """Newest run directory under ``videos/``, optionally filtered by task / kind prefix."""
    root = videos_root(log_dir)
    if not root.is_dir():
        return None
    runs: list[Path] = []
    task_dirs = [root / task] if task else [p for p in root.iterdir() if p.is_dir()]
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        for run_dir in task_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if kind is not None and not run_dir.name.startswith(f"{_safe_token(kind)}_"):
                continue
            runs.append(run_dir)
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def list_mp4s(folder: Path | str) -> list[Path]:
    path = Path(folder)
    if not path.is_dir():
        return []
    return sorted(path.glob("*.mp4"))


def find_named_mp4(root: Path | str, name: str) -> Path | None:
    """Prefer a direct child, else the newest recursive match."""
    base = Path(root)
    direct = base / name
    if direct.is_file():
        return direct
    matches = [p for p in base.rglob(name) if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)
