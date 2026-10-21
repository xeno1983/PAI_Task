"""Build a clean submission tree under ``提出/PickAndCarry``.

Keeps the report stack (teacher CPG → BC → curriculum PPO). Drops personal
paths, diagnostic scripts, and unused imitation variants (GAIL / DAgger /
ValueDICE). Re-run anytime; the destination is replaced.

    .venv\\Scripts\\python scripts\\export_submission.py
"""

from __future__ import annotations

import re
import shutil
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "提出" / "PickAndCarry"

# Modules / scripts that are not part of the submitted method.
DROP_FILES = {
    "gail.py",
    "train_gail.py",
    "valuedice.py",
    "train_valuedice.py",
    "dagger.py",
    "train_dagger.py",
    "diag_gail_vs_bc.py",
    "diag_trot_direction.py",
    "show_pose_heights.py",
    "export_submission.py",
}
DROP_TEST_PREFIXES = (
    "test_gail",
    "test_valuedice",
    "test_dagger",
)

COPY_DIRS = (
    "pick_and_carry",
    "scripts",
    "launcher",
    "assets",
    "tests",
    "ros",
    "docs",
)

# Internal planning notes — not part of the graded code pack.
DROP_DOC_NAMES = {
    "LOCOMOTION_PLAN.md",
    "REWARD_SCALES.md",
    "SIM_TO_REAL.md",
}

COPY_ROOT_FILES = (
    "requirements.txt",
    ".gitignore",
)

# Absolute / personal path fragments → portable replacements.
PATH_REPLACEMENTS = (
    (r"C:\\Users\\xeno1\\PAI_Task", "."),
    (r"C:/Users/xeno1/PAI_Task", "."),
    (r"/mnt/c/Users/xeno1/PAI_Task", "."),
    (r"C:\\Users\\xeno1\\Desktop\\起動スクリプト\\PickAndCarry", r".\\output"),
    (r"C:/Users/xeno1/Desktop/起動スクリプト/PickAndCarry", "./output"),
    (r"$env:USERPROFILE\\Desktop\\起動スクリプト\\PickAndCarry", r".\\output"),
    (r"$env:USERPROFILE/Desktop/起動スクリプト/PickAndCarry", "./output"),
    (r"xeno1", "student"),
)

# Full-line comments that are developer notes, not needed for grading.
COMMENT_DROP = re.compile(
    r"^\s*#.*(Footgun|this machine|Locked by scripts/diag|graded submission|"
    r"TODO|FIXME|XXX|hack|WIP|Cursor|Claude|GPT|あとで|とりあえず).*$",
    re.IGNORECASE,
)

TEXT_SUFFIXES = {".py", ".ps1", ".bat", ".md", ".txt", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini"}


def _rm_readonly(func, path, _exc):
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def should_skip(path: Path, *, relative: Path) -> bool:
    parts = set(relative.parts)
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return True
    if path.name in DROP_FILES:
        return True
    if path.suffix == ".pyc":
        return True
    if relative.parts[:1] == ("tests",) and path.name.startswith(DROP_TEST_PREFIXES):
        return True
    if path.suffix.lower() in {".pptx", ".pdf", ".mp4", ".html"} and "scripts" in parts:
        # Generated media / lecture copies stay out of the code pack.
        return True
    if path.name.startswith("PickAndCarry_学習条件"):
        return True
    if path.name.startswith("PickAndCarry_curves"):
        return True
    if path.name in DROP_DOC_NAMES:
        return True
    return False


def sanitize_text(text: str, *, suffix: str) -> str:
    out = text
    for old, new in PATH_REPLACEMENTS:
        out = out.replace(old, new)
    if suffix in {".py", ".ps1", ".bat", ".sh"}:
        lines = []
        for line in out.splitlines(keepends=True):
            if COMMENT_DROP.match(line.rstrip("\n\r")):
                continue
            lines.append(line)
        out = "".join(lines)
    # Soften leftover imitation advertising in copied docs / help strings.
    out = out.replace("BC / DAgger / ValueDICE", "BC")
    out = out.replace("BC/DAgger/ValueDICE", "BC")
    out = out.replace("DAgger / ValueDICE / GAIL", "BC")
    out = out.replace("teacher / BC / DAgger / ValueDICE / GAIL", "teacher / BC")
    return out


def copy_tree() -> None:
    if DEST.exists():
        shutil.rmtree(DEST, onerror=_rm_readonly)
    DEST.mkdir(parents=True)

    for name in COPY_ROOT_FILES:
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, DEST / name)

    for dirname in COPY_DIRS:
        src_root = ROOT / dirname
        if not src_root.is_dir():
            continue
        for src in src_root.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(ROOT)
            if should_skip(src, relative=rel):
                continue
            dst = DEST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() in TEXT_SUFFIXES or src.name in {
                "PickAndCarry_run.ps1",
                "PickAndCarry_run.bat",
                "display.sh",
            }:
                raw = src.read_text(encoding="utf-8", errors="replace")
                dst.write_text(sanitize_text(raw, suffix=src.suffix.lower()), encoding="utf-8")
            else:
                shutil.copy2(src, dst)


def patch_train_ppo(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = [
        (
            "Warm-start actor/critic. Imitation (bc/dagger/…) starts at iteration 0. ",
            "Warm-start actor/critic. Imitation (bc) starts at iteration 0. ",
        ),
        (
            "Aliases: bc / dagger / valuedice / gail / latest (logs/imitation + walk folders). ",
            "Aliases: bc / latest (logs/imitation + walk folders). ",
        ),
        (
            "Do not auto-load dagger/bc/valuedice/gail weights when --easy-walk and no --init-from.",
            "Do not auto-load bc weights when --easy-walk and no --init-from.",
        ),
        (
            'for name in ("dagger_trot.pt", "bc_trot.pt", "valuedice_trot.pt", "gail_trot.pt"):',
            'for name in ("bc_trot.pt",):',
        ),
        (
            '    elif "dagger_trot" in resume:\n'
            '        kind = "dagger"\n'
            '    elif "valuedice_trot" in resume:\n'
            '        kind = "valuedice"\n'
            '    elif "gail_trot" in resume:\n'
            '        kind = "gail"\n',
            "",
        ),
        (
            '"bc": ("bc_trot.pt",),\n'
            '        "bc_trot": ("bc_trot.pt",),\n'
            '        "dagger": ("dagger_trot.pt",),\n'
            '        "dagger_trot": ("dagger_trot.pt",),\n'
            '        "valuedice": ("valuedice_trot.pt",),\n'
            '        "valuedice_trot": ("valuedice_trot.pt",),\n'
            '        "gail": ("gail_trot.pt", "valuedice_trot.pt"),\n'
            '        "gail_trot": ("gail_trot.pt", "valuedice_trot.pt"),\n'
            '        "latest": ("dagger_trot.pt", "bc_trot.pt", "valuedice_trot.pt", "gail_trot.pt"),',
            '"bc": ("bc_trot.pt",),\n'
            '        "bc_trot": ("bc_trot.pt",),\n'
            '        "latest": ("bc_trot.pt",),',
        ),
        (
            '    if blob.get("dagger_trot"):\n'
            '        kind = "DAgger"\n'
            '    elif blob.get("valuedice_trot"):\n'
            '        kind = "ValueDICE"\n'
            '    elif blob.get("gail_trot"):\n'
            '        kind = "GAIL"\n'
            '    elif blob.get("bc_trot"):\n'
            '        kind = "BC trot"\n',
            '    if blob.get("bc_trot"):\n'
            '        kind = "BC trot"\n',
        ),
        (
            "        # Prefer DAgger / BC for walk (env PPO), then ValueDICE / legacy GAIL.\n",
            "        # Prefer BC warm-start when no PPO checkpoint exists.\n",
        ),
        (
            '"""Resolve ``--init-from`` path or alias (``bc``, ``dagger``, ``latest``, …).',
            '"""Resolve ``--init-from`` path or alias (``bc``, ``latest``).',
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def patch_video_paths(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "IMITATION_VIDEO_KINDS = frozenset({\"teacher\", \"bc\", \"dagger\", \"valuedice\", \"gail\"})",
        "IMITATION_VIDEO_KINDS = frozenset({\"teacher\", \"bc\"})",
    )
    path.write_text(text, encoding="utf-8")


def patch_test_video_paths(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'for kind in ("teacher", "bc", "dagger", "valuedice", "gail"):',
        'for kind in ("teacher", "bc"):',
    )
    text = re.sub(
        r'^\s*self\.assertEqual\(video_family_for_kind\("(dagger|valuedice|gail)"\), "imitation"\)\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text, encoding="utf-8")


def patch_colab_doc(path: Path) -> None:
    """Replace development COLAB notes with a short portable guide."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Colab での実行（任意）

提出手法はローカルと同じく **教師 CPG → BC → PPO** です。

1. このフォルダを zip して Colab にアップロードし展開する。
2. GPU ランタイムを選ぶ。
3. 依存関係を入れたあと、例えば次を実行する。

```python
!python -m pick_and_carry.train_bc --easy-walk --num-envs 8 --seconds 30 --epochs 50 --log-dir logs/imitation --arm z1
!python -m pick_and_carry.train_ppo --easy-walk --init-from logs/imitation/bc_trot.pt --num-envs 8 --iterations 50 --curriculum walk --arm z1 --log-dir logs/ppo_curriculum_walk_easy
```

詳細な引数はリポジトリ直下の `README.md` を参照してください。
""",
        encoding="utf-8",
    )


def patch_output_defaults(*paths: Path) -> None:
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            'Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"',
            'Path(__file__).resolve().parents[1] / "output"',
        )
        path.write_text(text, encoding="utf-8")


def patch_imitation_names() -> None:
    curriculum = DEST / "pick_and_carry" / "curriculum.py"
    if curriculum.is_file():
        text = curriculum.read_text(encoding="utf-8")
        text = text.replace(
            '_LEGACY_IMITATION_NAMES = (\n'
            '    "bc_trot.pt",\n'
            '    "dagger_trot.pt",\n'
            '    "valuedice_trot.pt",\n'
            '    "gail_trot.pt",\n'
            ')',
            '_LEGACY_IMITATION_NAMES = (\n'
            '    "bc_trot.pt",\n'
            ')',
        )
        text = text.replace(
            '"""Folders to search for ``bc_trot.pt`` / ``dagger_trot.pt`` / ``valuedice_trot.pt``.',
            '"""Folders to search for ``bc_trot.pt``.',
        )
        curriculum.write_text(text, encoding="utf-8")

    checkpoints = DEST / "pick_and_carry" / "checkpoints.py"
    if checkpoints.is_file():
        text = checkpoints.read_text(encoding="utf-8")
        text = text.replace(
            '_IMITATION_FLAGS = ("bc_trot", "dagger_trot", "valuedice_trot", "gail_trot")',
            '_IMITATION_FLAGS = ("bc_trot",)',
        )
        text = text.replace('"""True for BC / GAIL warm-start files."""', '"""True for BC warm-start files."""')
        checkpoints.write_text(text, encoding="utf-8")

    mlflow = DEST / "pick_and_carry" / "mlflow_log.py"
    if mlflow.is_file():
        text = mlflow.read_text(encoding="utf-8")
        text = re.sub(
            r"[ \t]*# BC / GAIL share the walk log dir with PPO; prefix so runs are easy to filter\.\n",
            "",
            text,
        )
        text = re.sub(
            r"[ \t]*if algo in \(\s*\"bc\"\s*,\s*\"gail\"\s*\):",
            '        if algo == "bc":',
            text,
        )
        text = re.sub(
            r"[ \t]*\"gail_coef\": info\.get\(\"gail_coef\"\),\n",
            "",
            text,
        )
        mlflow.write_text(text, encoding="utf-8")

    for rel in (
        "pick_and_carry/trot_teacher.py",
        "pick_and_carry/cpg.py",
        "pick_and_carry/video_paths.py",
        "scripts/make_presentation.py",
    ):
        path = DEST / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace("cloning and GAIL warm-start", "cloning warm-start")
        text = text.replace("``scripts/diag_trot_direction.py`` after any swing-sign edit.\n", "")
        text = text.replace("(needed for DAgger when the env may apply the student action instead).\n", "")
        text = text.replace("keep couple≈1 and entrain≈0 so flat-ground BC/GAIL stay rigid diagonal.", "keep couple≈1 and entrain≈0 for a rigid diagonal trot on flat ground.")
        text = text.replace("Imitation demos (teacher / BC / GAIL) live under", "Imitation demos (teacher / BC) live under")
        text = text.replace(
            "# Narrative order. Teacher/BC show only forward + left/right turn (no full reel, no GAIL).\n",
            "# Narrative order: teacher / BC forward and turns, then curriculum stages.\n",
        )
        text = text.replace(
            'hint = "メニュー 1 の模倣学習" if "trot" in name or name.startswith("bc_") or name.startswith("gail_") else "メニュー 2 でその段階を学習"',
            'hint = "メニュー 1 の模倣学習" if "trot" in name or name.startswith("bc_") else "メニュー 2 でその段階を学習"',
        )
        path.write_text(text, encoding="utf-8")


def patch_make_presentation(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'LAUNCHER = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"',
        'LAUNCHER = Path(__file__).resolve().parents[1] / "output"',
    )
    text = re.sub(
        r'^# Course-provided bookends.*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^# Submission target:.*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^# Final submission page picks.*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^# Do not dump every skill clip;.*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    # Drop GAIL alias rows from preferred aliases if present.
    text = re.sub(r'^\s*"gail_trot[^"]*".*\n', "", text, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


def patch_make_summary_slide(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"',
        'Path(__file__).resolve().parents[1] / "output"',
    )
    text = text.replace("BC・GAIL・PPO", "BC・PPO")
    path.write_text(text, encoding="utf-8")


def write_launcher() -> None:
    """Portable launcher: project root = parent of launcher/, output = ./output."""
    ps1 = r'''chcp.com 65001 > $null
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# launcher/ sits next to the project root in the submission tree.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectRoot "pick_and_carry"))) {
    $ProjectRoot = $PSScriptRoot
}
Set-Location $ProjectRoot
$Host.UI.RawUI.WindowTitle = "Pick-and-Carry"
$out = Join-Path $ProjectRoot "output"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host ".venv がありません。README の手順で環境を作ってから再実行してください。"
    Read-Host "キーを押すと閉じます"
    exit 1
}

Write-Host ""
Write-Host " Pick-and-Carry（提出用）"
Write-Host " プロジェクト: $(Get-Location)"
Write-Host " 出力: $out"
Write-Host ""
Write-Host "   1  模倣学習（トロット教師 → BC）"
Write-Host "   2  段階学習（歩行 / 到達 / 持ち運び / 箱運び / キャッチ）"
Write-Host "   3  推論（保存した重みを再生）"
Write-Host "   4  箱運びの途中経過を並べた比較動画"
Write-Host "   5  学習曲線を開く"
Write-Host "   6  プレゼン用の動画を作る"
Write-Host ""

function Get-LatestIteration {
    param([string]$Folder = "logs\ppo_z1")
    $dir = Join-Path (Get-Location) $Folder
    $best = 0
    if (Test-Path $dir) {
        foreach ($file in Get-ChildItem -Path $dir -Filter "ppo_*.pt" -ErrorAction SilentlyContinue) {
            if ($file.BaseName -match '^ppo_(\d+)$') {
                $value = [int]$Matches[1]
                if ($value -gt $best) { $best = $value }
            }
        }
    }
    return $best
}

function Ask-Int {
    param([string]$Prompt, [int]$Default)
    $raw = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    $parsed = 0
    if ([int]::TryParse($raw, [ref]$parsed) -and $parsed -gt 0) { return $parsed }
    Write-Host "正の整数を入れてください。既定の $Default を使います。"
    return $Default
}

function Get-BcPath {
    $candidates = @(
        "logs\imitation\bc_trot.pt",
        "logs\ppo_curriculum_walk_easy\bc_trot.pt",
        "logs\ppo_curriculum_walk\bc_trot.pt"
    )
    foreach ($rel in $candidates) {
        $p = Join-Path (Get-Location) $rel
        if (Test-Path $p) { return $p }
    }
    return $null
}

$choice = Read-Host "番号"
switch ($choice) {
    "1" {
        $logDir = "logs\imitation"
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        $seconds = Ask-Int "教師デモ秒数" 90
        $epochs = Ask-Int "BC epochs" 3000
        $envs = Ask-Int "並列環境数" 32
        Write-Host "トロット教師 → BC を実行します…"
        & $python -u -m pick_and_carry.train_bc --easy-walk --num-envs $envs --seconds $seconds --epochs $epochs --log-dir $logDir --arm z1
        Write-Host "完了。bc_trot.pt は $logDir にあります。"
    }
    "2" {
        Write-Host "段階: 1 歩行(簡単)  2 到達  3 持ち運び  4 箱運び  5 キャッチ"
        $stage = Read-Host "段階 [1]"
        if ([string]::IsNullOrWhiteSpace($stage)) { $stage = "1" }
        $iters = Ask-Int "学習回数" 1000
        $envs = Ask-Int "並列環境数" 64
        $map = @{
            "1" = @("walk", "logs\ppo_curriculum_walk_easy", $true)
            "2" = @("reach", "logs\ppo_curriculum_reach", $false)
            "3" = @("haul", "logs\ppo_curriculum_haul", $false)
            "4" = @("carry", "logs\ppo_z1", $false)
            "5" = @("catch", "logs\ppo_curriculum_catch", $false)
        }
        if (-not $map.ContainsKey($stage)) {
            Write-Host "未知の段階です。"
            break
        }
        $cur = $map[$stage][0]
        $logDir = $map[$stage][1]
        $easy = $map[$stage][2]
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        $args = @(
            "-u", "-m", "pick_and_carry.train_ppo",
            "--num-envs", "$envs", "--iterations", "$iters",
            "--num-steps", "128", "--save-interval", "20",
            "--log-dir", $logDir, "--curriculum", $cur, "--arm", "z1",
            "--resume", "latest"
        )
        if ($easy) {
            $args += "--easy-walk"
            $bc = Get-BcPath
            if ($null -ne $bc) {
                Write-Host "BC 重みから初期化: $bc"
                $args += @("--init-from", $bc)
            }
        }
        Write-Host ("起動: python " + ($args -join " "))
        & $python @args
    }
    "3" {
        Write-Host "段階: 1 歩行(簡単)  2 到達  3 持ち運び  4 箱運び  5 キャッチ"
        $stage = Read-Host "段階 [4]"
        if ([string]::IsNullOrWhiteSpace($stage)) { $stage = "4" }
        $map = @{
            "1" = @("walk", "logs\ppo_curriculum_walk_easy", "PickAndCarry_walk.mp4", $true)
            "2" = @("reach", "logs\ppo_curriculum_reach", "PickAndCarry_reach.mp4", $false)
            "3" = @("haul", "logs\ppo_curriculum_haul", "PickAndCarry_haul.mp4", $false)
            "4" = @("carry", "logs\ppo_z1", "PickAndCarry_z1.mp4", $false)
            "5" = @("catch", "logs\ppo_curriculum_catch", "PickAndCarry_catch.mp4", $false)
        }
        if (-not $map.ContainsKey($stage)) {
            Write-Host "未知の段階です。"
            break
        }
        $cur = $map[$stage][0]
        $logDir = $map[$stage][1]
        $name = $map[$stage][2]
        $easy = $map[$stage][3]
        $record = Join-Path $out $name
        $seconds = Ask-Int "録画秒数" 20
        $play = @(
            "-u", "-m", "pick_and_carry.train_ppo",
            "--play", "--resume", "latest", "--num-envs", "1",
            "--seconds", "$seconds", "--record", $record,
            "--log-dir", $logDir, "--curriculum", $cur, "--arm", "z1"
        )
        if ($easy) { $play += "--easy-walk" }
        & $python @play
        if (Test-Path $record) { Start-Process $record }
    }
    "4" {
        $logDir = "logs\ppo_z1"
        $record = Join-Path $out "PickAndCarry_z1_compare.mp4"
        & $python -u scripts\compare_snapshots.py --log-dir $logDir --seconds 16 --panels 4 --output $record --arm z1
        if (Test-Path $record) { Start-Process $record }
    }
    "5" {
        $html = Join-Path $out "PickAndCarry_curves.html"
        & $python -u scripts\plot_curves.py --log-dir logs\ppo_z1 --output $html
        if (Test-Path $html) { Start-Process $html }
    }
    "6" {
        $pptx = Join-Path $out "PickAndCarry_学習条件.pptx"
        if (-not (Test-Path $pptx)) {
            Write-Host "スライドを先に生成します…"
            & $python -u scripts\make_summary_slide.py
        }
        & $python -u scripts\make_presentation.py --pptx $pptx --videos $out --output (Join-Path $out "PickAndCarry_presentation.mp4")
    }
    default {
        Write-Host "1〜6 を選んでください。"
    }
}

Write-Host ""
Read-Host "キーを押すと閉じます"
'''
    bat = r'''@echo off
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0PickAndCarry_run.ps1"
if errorlevel 1 (
    echo.
    echo The launcher stopped. This window stays open so you can read the message above.
    pause
)
'''
    (DEST / "launcher").mkdir(parents=True, exist_ok=True)
    (DEST / "launcher" / "PickAndCarry_run.ps1").write_text(ps1, encoding="utf-8")
    (DEST / "launcher" / "PickAndCarry_run.bat").write_text(bat, encoding="utf-8")
    # Convenience copies at project root.
    root_bat = (
        "@echo off\n"
        "cd /d \"%~dp0\"\n"
        "powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0launcher\\PickAndCarry_run.ps1\"\n"
        "if errorlevel 1 (\n"
        "    echo.\n"
        "    echo The launcher stopped. This window stays open so you can read the message above.\n"
        "    pause\n"
        ")\n"
    )
    (DEST / "PickAndCarry_run.bat").write_text(root_bat, encoding="utf-8")


def write_readme() -> None:
    readme = """# Go2 Pick-and-Carry（提出用）

Unitree Go2 + Z1 で、不整地を含むコース上の箱運びを学習する環境です。
提出手法は **トロット教師（CPG）→ 行動クローニング（BC）→ カリキュラム PPO** です。

## 構成

| パス | 内容 |
| --- | --- |
| `pick_and_carry/` | 環境・報酬・観測・CPG/IK・BC・PPO |
| `scripts/` | 起動補助、スライド／提出動画の生成 |
| `launcher/` | メニュー起動（`PickAndCarry_run.bat`） |
| `tests/` | 単体テスト（Genesis なしでも一部実行可） |
| `assets/urdf/` | Z1 付き簡易 URDF |
| `output/` | 実行時に生成（動画・スライド・曲線） |

## 環境構築（Windows）

```powershell
cd <このフォルダ>
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -U pip
.\\.venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
.\\.venv\\Scripts\\python.exe scripts\\smoke_local.py
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests
```

## 起動

`PickAndCarry_run.bat` または `launcher\\PickAndCarry_run.bat` を実行します。

1. 模倣学習（教師 → BC）
2. 段階学習（歩行 → 到達 → 持ち運び → 箱運び → キャッチ）
3. 推論・録画
4. 比較動画
5. 学習曲線
6. プレゼン動画

コマンド例:

```powershell
# BC
.\\.venv\\Scripts\\python.exe -m pick_and_carry.train_bc --easy-walk --num-envs 32 --seconds 90 --epochs 3000 --log-dir logs\\imitation --arm z1
# 歩行 PPO（BC から）
.\\.venv\\Scripts\\python.exe -m pick_and_carry.train_ppo --easy-walk --init-from logs\\imitation\\bc_trot.pt --num-envs 64 --iterations 1000 --curriculum walk --arm z1 --log-dir logs\\ppo_curriculum_walk_easy
```

## 補足

- 画面表示は使わず、結果は `output/` の mp4 で確認する想定です。
- ROS 2 連携は任意です（`ros/`）。必須ではありません。
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "提出" / "README.md").write_text(
        "このフォルダの `PickAndCarry/` が提出用コード一式です。\n"
        "個人パス・診断スクリプト・未使用の模倣手法（GAIL / DAgger / ValueDICE）を除いてあります。\n"
        "再生成: `python scripts/export_submission.py`\n",
        encoding="utf-8",
    )


def scrub_remaining_personal(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"display.sh"}:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    cleaned = sanitize_text(text, suffix=path.suffix.lower())
    if cleaned != text:
        path.write_text(cleaned, encoding="utf-8")


def main() -> None:
    print(f"提出ツリーを作成: {DEST}")
    copy_tree()
    patch_train_ppo(DEST / "pick_and_carry" / "train_ppo.py")
    patch_video_paths(DEST / "pick_and_carry" / "video_paths.py")
    patch_test_video_paths(DEST / "tests" / "test_video_paths.py")
    patch_make_presentation(DEST / "scripts" / "make_presentation.py")
    patch_make_summary_slide(DEST / "scripts" / "make_summary_slide.py")
    patch_colab_doc(DEST / "docs" / "COLAB.md")
    patch_output_defaults(
        DEST / "scripts" / "compare_snapshots.py",
        DEST / "scripts" / "plot_curves.py",
        DEST / "scripts" / "infer_gui.py",
        DEST / "scripts" / "smoke_local.py",
    )
    patch_imitation_names()
    # Drop leftover copies of the heavy development launcher under scripts/.
    for leftover in (
        DEST / "scripts" / "PickAndCarry_run.ps1",
        DEST / "scripts" / "PickAndCarry_run.bat",
    ):
        if leftover.is_file():
            leftover.unlink()
    write_launcher()
    write_readme()
    for path in DEST.rglob("*"):
        if path.is_file():
            scrub_remaining_personal(path)
    # Ensure output dir exists as a placeholder; drop any accidental runtime dirs.
    for runtime in ("logs", ".venv", "__pycache__"):
        junk = DEST / runtime
        if junk.exists():
            shutil.rmtree(junk, onerror=_rm_readonly)
    (DEST / "output").mkdir(exist_ok=True)
    (DEST / "output" / ".gitkeep").write_text("", encoding="utf-8")
    n = sum(1 for p in DEST.rglob("*") if p.is_file())
    print(f"完了: {n} ファイル → {DEST}")


if __name__ == "__main__":
    main()
