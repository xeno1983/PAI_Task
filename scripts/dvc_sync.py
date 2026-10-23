"""Publish / push / pull curated training artifacts via DVC + MinIO.

Does not touch the live ``logs/`` tree during training. After a stage finishes:

    .\\scripts\\load_pai_env.ps1
    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py publish --log-dir logs\\ppo_z1 --select best
    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py push

Play the curated best weight:

    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py pull
    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py resolve --name ppo_z1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "checkpoints"
CKPT_RE = re.compile(r"^(?:ppo|grpo|dr_grpo|sym_ppo)_(\d+)\.pt$")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def _dvc_bin() -> str:
    local = ROOT / ".venv" / "Scripts" / "dvc.exe"
    if local.is_file():
        return str(local)
    return "dvc"


def _stage_name(log_dir: Path) -> str:
    return log_dir.resolve().name


def _load_metrics(log_dir: Path) -> list[dict[str, Any]]:
    path = log_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _available_ckpts(log_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in log_dir.glob("*.pt"):
        match = CKPT_RE.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def _nearest_ckpt(ckpts: dict[int, Path], iteration: int) -> tuple[int, Path] | None:
    if not ckpts:
        return None
    best_iter = min(ckpts, key=lambda value: abs(value - iteration))
    return best_iter, ckpts[best_iter]


def _score_fn_for(log_dir: Path) -> Callable[[dict[str, Any]], float]:
    name = log_dir.name.lower()

    def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    if "walk" in name:
        return lambda r: 0.5 * _f(r, "tracking") + 0.3 * _f(r, "walk_score") + 0.2 * _f(r, "gait")
    if "reach" in name:
        return lambda r: 0.55 * _f(r, "approach") + 0.45 * _f(r, "reward")
    if "catch" in name:
        return lambda r: 0.7 * _f(r, "success") + 0.3 * _f(r, "reward")
    if "z1" in name or "carry" in name or "haul" in name:
        return lambda r: (
            0.40 * _f(r, "grasp")
            + 0.40 * (1.0 / (0.05 + _f(r, "ee_dist", 1.0)))
            + 0.20 * _f(r, "reward")
        )
    return lambda r: _f(r, "reward")


def _latest_ckpt(log_dir: Path) -> tuple[int, Path] | None:
    ckpts = _available_ckpts(log_dir)
    if not ckpts:
        return None
    iteration = max(ckpts)
    return iteration, ckpts[iteration]


def _best_ckpt(log_dir: Path) -> tuple[int, Path, float, dict[str, Any]] | None:
    """Pick the checkpoint nearest to the highest-scoring metrics row."""
    ckpts = _available_ckpts(log_dir)
    if not ckpts:
        return None
    rows = _load_metrics(log_dir)
    if not rows:
        latest = _latest_ckpt(log_dir)
        if latest is None:
            return None
        iteration, path = latest
        return iteration, path, float("-inf"), {"iteration": iteration, "note": "no metrics; used latest"}

    score_fn = _score_fn_for(log_dir)
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for row in rows:
        try:
            iteration = int(row["iteration"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            score = float(score_fn(row))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        ranked.append((score, iteration, row))
    if not ranked:
        latest = _latest_ckpt(log_dir)
        if latest is None:
            return None
        iteration, path = latest
        return iteration, path, float("-inf"), {"iteration": iteration, "note": "unscored; used latest"}

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score, metric_iter, row = ranked[0]
    hit = _nearest_ckpt(ckpts, metric_iter)
    if hit is None:
        return None
    ckpt_iter, path = hit
    return ckpt_iter, path, score, row


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def publish(
    log_dir: Path,
    *,
    name: str | None = None,
    keep: int = 1,
    select: str = "best",
    push_after: bool = False,
) -> Path:
    """Copy selected checkpoint(s) + metrics into ``artifacts/checkpoints/<name>/``."""
    log_dir = log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"log-dir がありません: {log_dir}")
    stage = name or _stage_name(log_dir)
    dest = ARTIFACTS / stage
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    copied: list[str] = []
    selection: dict[str, Any] = {"mode": select}

    if select == "latest":
        latest = _latest_ckpt(log_dir)
        if latest is None:
            raise SystemExit(f"{log_dir} に ppo_*.pt がありません。")
        iteration, path = latest
        selection.update({"iteration": iteration, "checkpoint": path.name, "score": None})
        chosen = [path]
        if keep > 1:
            ranked = sorted(_available_ckpts(log_dir).items(), reverse=True)
            chosen = [item[1] for item in ranked[:keep]]
    elif select == "best":
        best = _best_ckpt(log_dir)
        if best is None:
            raise SystemExit(f"{log_dir} に publish できる重みがありません。")
        iteration, path, score, row = best
        selection.update(
            {
                "iteration": iteration,
                "checkpoint": path.name,
                "score": None if score == float("-inf") else score,
                "metric_iteration": int(row.get("iteration", iteration)),
                "metrics": {
                    key: row[key]
                    for key in (
                        "reward",
                        "tracking",
                        "walk_score",
                        "gait",
                        "approach",
                        "grasp",
                        "ee_dist",
                        "success",
                    )
                    if key in row
                },
            }
        )
        chosen = [path]
        print(
            f"best select: {path.name} (iter={iteration}, score={selection['score']})",
            flush=True,
        )
    else:
        raise SystemExit(f"unknown --select: {select} (use best|latest)")

    for path in chosen:
        if _copy_if_exists(path, dest / path.name):
            copied.append(path.name)
    # Stable name for play / restore consumers.
    primary = chosen[0]
    shutil.copy2(primary, dest / "best.pt")
    copied.append("best.pt")

    for extra in (
        "bc_trot.pt",
        "dagger_trot.pt",
        "valuedice_trot.pt",
        "gail_trot.pt",
        "metrics.jsonl",
        "run.json",
    ):
        if _copy_if_exists(log_dir / extra, dest / extra):
            copied.append(extra)

    meta = {
        "source_log_dir": str(log_dir),
        "stage": stage,
        "selection": selection,
        "files": copied,
    }
    (dest / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not copied:
        raise SystemExit(f"{log_dir} に publish できる重み / メトリクスがありません。")

    dvc = _dvc_bin()
    _run([dvc, "add", str(dest.relative_to(ROOT))])
    print(f"published {len(copied)} file(s) → {dest.relative_to(ROOT)}", flush=True)
    if push_after:
        push()
    return dest


def restore(log_dir: Path, *, name: str | None = None) -> None:
    """Copy curated artifacts back into a training log folder."""
    stage = name or _stage_name(log_dir)
    src = ARTIFACTS / stage
    if not src.is_dir():
        raise SystemExit(f"artifacts がありません（先に pull）: {src}")
    log_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    for path in src.iterdir():
        if path.name == "manifest.json" or not path.is_file():
            continue
        shutil.copy2(path, log_dir / path.name)
        restored += 1
        print(f"restore {path.name} → {log_dir / path.name}", flush=True)
    if restored == 0:
        raise SystemExit(f"{src} が空です。")
    print(f"restored {restored} file(s) into {log_dir}", flush=True)


def resolve(name: str, *, prefer: str = "best") -> Path:
    """Return the curated checkpoint path for play / --resume."""
    src = ARTIFACTS / name
    if not src.is_dir():
        raise SystemExit(f"artifacts がありません（先に pull）: {src}")
    best = src / "best.pt"
    if prefer == "best" and best.is_file():
        print(best.resolve(), flush=True)
        return best.resolve()
    manifest = src / "manifest.json"
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        ckpt_name = (meta.get("selection") or {}).get("checkpoint")
        if ckpt_name and (src / ckpt_name).is_file():
            path = (src / ckpt_name).resolve()
            print(path, flush=True)
            return path
    ranked = sorted(
        (path for path in src.glob("*.pt") if CKPT_RE.match(path.name)),
        key=lambda path: int(CKPT_RE.match(path.name).group(1)),  # type: ignore[union-attr]
        reverse=True,
    )
    if not ranked:
        raise SystemExit(f"{src} に推論用 .pt がありません。")
    path = ranked[0].resolve()
    print(path, flush=True)
    return path


def push() -> None:
    _run([_dvc_bin(), "push"])


def pull() -> None:
    _run([_dvc_bin(), "pull"])


def status() -> None:
    _run([_dvc_bin(), "status"], check=False)
    if not ARTIFACTS.is_dir():
        print("local curated stages: (none)", flush=True)
        return
    stages = sorted(path for path in ARTIFACTS.iterdir() if path.is_dir())
    if not stages:
        print("local curated stages: (none)", flush=True)
        return
    print("local curated stages:", flush=True)
    for stage in stages:
        manifest = stage / "manifest.json"
        best = stage / "best.pt"
        detail = ""
        if manifest.is_file():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            sel = meta.get("selection") or {}
            detail = (
                f"  select={sel.get('mode', '?')} iter={sel.get('iteration', '?')} "
                f"score={sel.get('score', '?')} ckpt={sel.get('checkpoint', '?')}"
            )
        flag = "best.pt" if best.is_file() else "no-best.pt"
        print(f"  {stage.name}  [{flag}]{detail}", flush=True)


def ensure_minio_env() -> None:
    endpoint = os.environ.get("PAI_MINIO_ENDPOINT") or os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    if not endpoint:
        print(
            "警告: PAI_MINIO_ENDPOINT が未設定です。先に . .\\scripts\\load_pai_env.ps1 を実行してください。",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DVC checkpoint publish / sync helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pub = sub.add_parser("publish", help="Copy weights from a log-dir into artifacts/ and dvc add.")
    p_pub.add_argument("--log-dir", type=Path, required=True)
    p_pub.add_argument("--name", type=str, default=None, help="Artifact folder name (default: log-dir name).")
    p_pub.add_argument("--keep", type=int, default=1, help="How many newest ppo_*.pt to keep (select=latest only).")
    p_pub.add_argument(
        "--select",
        choices=("best", "latest"),
        default="best",
        help="best=metrics peak (default), latest=highest iteration.",
    )
    p_pub.add_argument("--push", action="store_true", help="dvc push after publish.")

    p_res = sub.add_parser("restore", help="Copy artifacts back into a log-dir for --resume.")
    p_res.add_argument("--log-dir", type=Path, required=True)
    p_res.add_argument("--name", type=str, default=None)

    p_resolve = sub.add_parser("resolve", help="Print curated checkpoint path for --play / --resume.")
    p_resolve.add_argument("--name", type=str, required=True, help="Artifact folder under artifacts/checkpoints/.")
    p_resolve.add_argument("--prefer", choices=("best", "manifest"), default="best")

    sub.add_parser("push", help="dvc push to MinIO.")
    sub.add_parser("pull", help="dvc pull from MinIO.")
    sub.add_parser("status", help="dvc status + list local curated stages.")
    return parser.parse_args()


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    ensure_minio_env()
    if args.cmd == "publish":
        publish(
            args.log_dir,
            name=args.name,
            keep=max(1, args.keep),
            select=args.select,
            push_after=bool(args.push),
        )
    elif args.cmd == "restore":
        restore(args.log_dir, name=args.name)
    elif args.cmd == "resolve":
        resolve(args.name, prefer=args.prefer)
    elif args.cmd == "push":
        push()
    elif args.cmd == "pull":
        pull()
    elif args.cmd == "status":
        status()
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
