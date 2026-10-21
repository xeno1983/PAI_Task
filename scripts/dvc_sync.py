"""Publish / push / pull curated training artifacts via DVC + MinIO.

Does not touch the live ``logs/`` tree during training. After a stage finishes:

    .\\scripts\\load_pai_env.ps1
    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py publish --log-dir logs\\ppo_z1
    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py push

On another machine:

    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py pull
    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py restore --log-dir logs\\ppo_z1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "checkpoints"


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


def _latest_ppo(log_dir: Path) -> Path | None:
    best: Path | None = None
    best_iter = -1
    for path in log_dir.glob("ppo_*.pt"):
        stem = path.stem
        if stem.startswith("ppo_") and stem[4:].isdigit():
            value = int(stem[4:])
            if value >= best_iter:
                best_iter = value
                best = path
    return best


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def publish(log_dir: Path, *, name: str | None = None, keep: int = 1) -> Path:
    """Copy latest checkpoint(s) + metrics into ``artifacts/checkpoints/<name>/``."""
    log_dir = log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"log-dir がありません: {log_dir}")
    stage = name or _stage_name(log_dir)
    dest = ARTIFACTS / stage
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    copied: list[str] = []
    latest = _latest_ppo(log_dir)
    if latest is not None:
        _copy_if_exists(latest, dest / latest.name)
        copied.append(latest.name)
        if keep > 1:
            ranked = sorted(
                (
                    path
                    for path in log_dir.glob("ppo_*.pt")
                    if path.stem.startswith("ppo_") and path.stem[4:].isdigit()
                ),
                key=lambda path: int(path.stem[4:]),
                reverse=True,
            )
            for path in ranked[1:keep]:
                if _copy_if_exists(path, dest / path.name):
                    copied.append(path.name)

    for name in (
        "bc_trot.pt",
        "dagger_trot.pt",
        "valuedice_trot.pt",
        "gail_trot.pt",
        "metrics.jsonl",
        "run.json",
    ):
        if _copy_if_exists(log_dir / name, dest / name):
            copied.append(name)

    meta = {
        "source_log_dir": str(log_dir),
        "stage": stage,
        "files": copied,
    }
    (dest / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not copied:
        raise SystemExit(f"{log_dir} に publish できる重み / メトリクスがありません。")

    dvc = _dvc_bin()
    _run([dvc, "add", str(dest.relative_to(ROOT))])
    print(f"published {len(copied)} file(s) → {dest.relative_to(ROOT)}", flush=True)
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


def push() -> None:
    _run([_dvc_bin(), "push"])


def pull() -> None:
    _run([_dvc_bin(), "pull"])


def status() -> None:
    _run([_dvc_bin(), "status"], check=False)
    if ARTIFACTS.is_dir():
        stages = sorted(path.name for path in ARTIFACTS.iterdir() if path.is_dir())
        print("local curated stages:", ", ".join(stages) if stages else "(none)", flush=True)


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

    p_pub = sub.add_parser("publish", help="Copy latest weights from a log-dir into artifacts/ and dvc add.")
    p_pub.add_argument("--log-dir", type=Path, required=True)
    p_pub.add_argument("--name", type=str, default=None, help="Artifact folder name (default: log-dir name).")
    p_pub.add_argument("--keep", type=int, default=1, help="How many newest ppo_*.pt to keep.")

    p_res = sub.add_parser("restore", help="Copy artifacts back into a log-dir for --resume.")
    p_res.add_argument("--log-dir", type=Path, required=True)
    p_res.add_argument("--name", type=str, default=None)

    sub.add_parser("push", help="dvc push to MinIO.")
    sub.add_parser("pull", help="dvc pull from MinIO.")
    sub.add_parser("status", help="dvc status + list local curated stages.")
    return parser.parse_args()


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    ensure_minio_env()
    if args.cmd == "publish":
        publish(args.log_dir, name=args.name, keep=max(1, args.keep))
    elif args.cmd == "restore":
        restore(args.log_dir, name=args.name)
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
