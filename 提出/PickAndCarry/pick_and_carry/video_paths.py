"""Video output layout: ``{log_dir}/videos/[imitation/]{task}/{run_id}/*.mp4``.

``task`` names the curriculum (e.g. ``walk_easy``, ``carry``, ``catch_slow``).
Imitation demos (teacher / BC) live under
``videos/imitation/`` so they stay separate from RL play clips under ``videos/``.
``run_id`` encodes which model produced the clip and training progress, plus a
timestamp so repeated runs do not overwrite each other.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Kind prefixes that belong under videos/imitation/ (not next to PPO play clips).
IMITATION_VIDEO_KINDS = frozenset({"teacher", "bc"})
IMITATION_VIDEO_FAMILY = "imitation"


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


def video_family_for_kind(kind: str | None) -> str | None:
    """Return ``imitation`` for teacher/BC/… kinds, else ``None`` (RL play tree)."""
    if kind is None:
        return None
    token = _safe_token(kind).split("_", 1)[0]
    if token in IMITATION_VIDEO_KINDS or _safe_token(kind) in IMITATION_VIDEO_KINDS:
        return IMITATION_VIDEO_FAMILY
    return None


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


def videos_root(log_dir: Path | str, *, family: str | None = None) -> Path:
    """``{log_dir}/videos`` or ``{log_dir}/videos/imitation`` when ``family`` is set."""
    root = Path(log_dir) / "videos"
    if family:
        return root / _safe_token(family)
    return root


def is_under_videos_tree(path: Path | str, log_dir: Path | str) -> bool:
    """True if ``path`` sits anywhere under ``{log_dir}/videos/`` (any family)."""
    try:
        Path(path).resolve().relative_to((Path(log_dir) / "videos").resolve())
        return True
    except (ValueError, OSError):
        return False


def make_video_run_dir(
    log_dir: Path | str,
    *,
    task: str,
    kind: str,
    iteration: int | None = None,
    epochs: int | None = None,
    stamp: str | None = None,
    extra: dict | None = None,
    family: str | None = None,
) -> Path:
    """Create ``videos/[imitation/]{task}/{run_id}/`` and write a small ``run.json``."""
    if family is None:
        family = video_family_for_kind(kind)
    run_id = make_run_id(kind, iteration=iteration, epochs=epochs, stamp=stamp)
    path = (videos_root(Path(log_dir).resolve(), family=family) / _safe_token(task) / run_id).resolve()
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": task,
        "kind": kind,
        "family": family,
        "run_id": run_id,
        "iteration": iteration,
        "epochs": epochs,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        manifest.update(extra)
    (path / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _run_kind_prefix(run_dir: Path) -> str:
    return run_dir.name.split("_", 1)[0]


def latest_video_run(
    log_dir: Path | str,
    *,
    task: str | None = None,
    kind: str | None = None,
    family: str | None = None,
) -> Path | None:
    """Newest run directory under ``videos/`` (and ``videos/imitation/``), optionally filtered."""
    if family is None and kind is not None:
        family = video_family_for_kind(kind)
    base = Path(log_dir) / "videos"
    if not base.is_dir():
        return None
    roots: list[Path] = []
    if family is not None:
        roots.append(videos_root(log_dir, family=family))
    else:
        roots.append(base)
        imit = base / IMITATION_VIDEO_FAMILY
        if imit.is_dir():
            roots.append(imit)
    runs: list[Path] = []
    base_resolved = base.resolve()
    for root in roots:
        if not root.is_dir():
            continue
        if task:
            task_dirs = [root / task]
        else:
            task_dirs = [p for p in root.iterdir() if p.is_dir()]
            if root.resolve() == base_resolved:
                task_dirs = [p for p in task_dirs if p.name != IMITATION_VIDEO_FAMILY]
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


def migrate_legacy_imitation_videos(log_dir: Path | str) -> list[Path]:
    """Move flat ``videos/<task>/<imitation_kind>_…`` runs under ``videos/imitation/<task>/``.

    Safe no-op when already nested. Returns destination run directories that were moved.
    """
    log_dir = Path(log_dir)
    base = log_dir / "videos"
    if not base.is_dir():
        return []
    moved: list[Path] = []
    for task_dir in list(base.iterdir()):
        if not task_dir.is_dir() or task_dir.name == IMITATION_VIDEO_FAMILY:
            continue
        for run_dir in list(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if video_family_for_kind(_run_kind_prefix(run_dir)) != IMITATION_VIDEO_FAMILY:
                continue
            dest_parent = videos_root(log_dir, family=IMITATION_VIDEO_FAMILY) / task_dir.name
            dest_parent.mkdir(parents=True, exist_ok=True)
            dest = dest_parent / run_dir.name
            if dest.exists():
                # Keep the newer tree; drop the leftover flat copy.
                shutil.rmtree(run_dir, ignore_errors=True)
                print(f"旧デモ配置の重複を削除: {run_dir}", flush=True)
                continue
            run_dir.replace(dest)
            moved.append(dest)
            print(f"デモ動画を移動: {run_dir} → {dest}", flush=True)
        # Remove empty task dirs left behind under videos/ (not imitation/).
        try:
            if task_dir.is_dir() and not any(task_dir.iterdir()):
                task_dir.rmdir()
        except OSError:
            pass
    return moved
