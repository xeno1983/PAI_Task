"""Pack a curated DVC checkpoint (best.pt) into a Mender Artifact for OTA.

Requires the ``mender-artifact`` CLI on PATH (Linux/macOS binary, or WSL on Windows).

Example:

    .\\scripts\\load_pai_env.ps1
    .\\.venv\\Scripts\\python.exe scripts\\dvc_sync.py publish --log-dir logs\\ppo_z1 --select best
    .\\.venv\\Scripts\\python.exe scripts\\mender_pack_model.py --name ppo_z1 --device-type go2-z1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "checkpoints"
OUT_DIR = ROOT / "artifacts" / "mender"


def _find_mender_artifact() -> str:
    exe = shutil.which("mender-artifact")
    if exe:
        return exe
    # Common Windows / local drop locations.
    for candidate in (
        ROOT / "tools" / "mender-artifact.exe",
        ROOT / "tools" / "mender-artifact",
        Path.home() / "bin" / "mender-artifact",
    ):
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        "mender-artifact が見つかりません。\n"
        "  https://docs.mender.io/downloads から取得し PATH か tools/ に置いてください。\n"
        "  Windows では WSL 上の mender-artifact でも可。"
    )


def _stage_dir(name: str) -> Path:
    path = ARTIFACTS / name
    if not path.is_dir():
        raise SystemExit(f"artifacts がありません（先に dvc_sync publish / pull）: {path}")
    best = path / "best.pt"
    if not best.is_file():
        raise SystemExit(f"best.pt がありません: {best}")
    return path


def pack(
    *,
    name: str,
    device_type: str,
    artifact_name: str | None,
    output: Path | None,
    software_version: str | None,
) -> Path:
    stage = _stage_dir(name)
    manifest_path = stage / "manifest.json"
    selection: dict = {}
    if manifest_path.is_file():
        selection = json.loads(manifest_path.read_text(encoding="utf-8")).get("selection") or {}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    iter_tag = selection.get("iteration", "na")
    art_name = artifact_name or f"pai-{name}-iter{iter_tag}-{stamp}"
    version = software_version or f"iter{iter_tag}-{stamp}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output or (OUT_DIR / f"{art_name}.mender")

    with tempfile.TemporaryDirectory(prefix="pai-mender-") as tmp:
        tmp_path = Path(tmp)
        payload = tmp_path / "payload"
        payload.mkdir()
        shutil.copy2(stage / "best.pt", payload / "best.pt")
        for extra in ("manifest.json", "metrics.jsonl", "run.json"):
            src = stage / extra
            if src.is_file():
                shutil.copy2(src, payload / extra)

        meta = {
            "stage": name,
            "selection": selection,
            "built_at": stamp,
            "source": str(stage.relative_to(ROOT)).replace("\\", "/"),
        }
        meta_file = tmp_path / "meta-data.json"
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        cmd = [
            _find_mender_artifact(),
            "write",
            "module-image",
            "-T",
            "pai-model",
            "-n",
            art_name,
            "-t",
            device_type,
            "-o",
            str(out),
            "--software-name",
            f"pai-model-{name}",
            "--software-version",
            version,
            "-m",
            str(meta_file),
            "-f",
            str(payload / "best.pt"),
        ]
        for extra in ("manifest.json", "metrics.jsonl", "run.json"):
            path = payload / extra
            if path.is_file():
                cmd.extend(["-f", str(path)])

        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    side = out.with_suffix(out.suffix + ".json")
    side.write_text(
        json.dumps(
            {
                "artifact": str(out.relative_to(ROOT)).replace("\\", "/"),
                "stage": name,
                "device_type": device_type,
                "artifact_name": art_name,
                "software_version": version,
                "selection": selection,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    print(f"meta  {side}", flush=True)
    print(
        "次: Hosted Mender / 自前サーバにアップロードし、device_type="
        f"{device_type} の端末へデプロイ。端末には device/mender/modules/pai-model を入れておく。",
        flush=True,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack DVC best.pt into a Mender Artifact.")
    parser.add_argument("--name", required=True, help="artifacts/checkpoints/<name> (e.g. ppo_curriculum_walk)")
    parser.add_argument(
        "--device-type",
        default=os.environ.get("PAI_MENDER_DEVICE_TYPE", "go2-z1"),
        help="Mender device_type compatible string (default: go2-z1).",
    )
    parser.add_argument("--artifact-name", default=None, help="Override Artifact name (-n).")
    parser.add_argument("--software-version", default=None, help="Override --software-version.")
    parser.add_argument("--output", type=Path, default=None, help="Output .mender path.")
    return parser.parse_args()


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    pack(
        name=args.name,
        device_type=args.device_type,
        artifact_name=args.artifact_name,
        output=args.output,
        software_version=args.software_version,
    )


if __name__ == "__main__":
    main()
