"""Join the condition slides and recorded mp4s into one presentation video.

    .venv\\Scripts\\python scripts\\make_presentation.py

Slides come from the PowerPoint in the launcher folder. Each slide stays on
screen, then the recordings play in a fixed order. The result is
PickAndCarry_presentation.mp4 in that same folder.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve().parents[1] / "output"
WIDTH = 1280
HEIGHT = 720
FPS = 25
INK = (28, 40, 44)
WHITE = (255, 255, 255)
OP_PATH = ROOT / "PAI応用1_最終課題_OP.mp4"
ED_PATH = ROOT / "PAI応用1_最終課題_ED.mp4"
BOOKEND_NAMES = frozenset({OP_PATH.name, ED_PATH.name})

# Narrative order: teacher / BC forward and turns, then curriculum stages.
PREFERRED = (
    "PickAndCarry_smoke.mp4",
    "PickAndCarry_z1_smoke.mp4",
    "trot_teacher_forward.mp4",
    "trot_teacher_turn_left.mp4",
    "trot_teacher_turn_right.mp4",
    "bc_trot_forward.mp4",
    "bc_trot_turn_left.mp4",
    "bc_trot_turn_right.mp4",
    "PickAndCarry_walk.mp4",
    "PickAndCarry_reach.mp4",
    "PickAndCarry_haul.mp4",
    "PickAndCarry_z1.mp4",
    "PickAndCarry_catch_static.mp4",
    "PickAndCarry_catch_slow.mp4",
    "PickAndCarry_catch_random.mp4",
    "PickAndCarry_catch_fly.mp4",
    "PickAndCarry_catch_noisy.mp4",
    "PickAndCarry_catch.mp4",
    "PickAndCarry_compare.mp4",
    "PickAndCarry_z1_compare.mp4",
)
TITLES = {
    "PickAndCarry_smoke.mp4": ("起動確認", "ロボットが足場に立つかを見ます"),
    "PickAndCarry_z1_smoke.mp4": ("起動確認（Z1）", "軽い腕でも足場に立つかを見ます"),
    "trot_teacher_forward.mp4": ("教師・前進", "CPG 教師の前進です"),
    "trot_teacher_turn_left.mp4": ("教師・左旋回", "CPG 教師の左旋回です"),
    "trot_teacher_turn_right.mp4": ("教師・右旋回", "CPG 教師の右旋回です"),
    "bc_trot_forward.mp4": ("BC・前進", "行動クローニングの前進です"),
    "bc_trot_turn_left.mp4": ("BC・左旋回", "行動クローニングの左旋回です"),
    "bc_trot_turn_right.mp4": ("BC・右旋回", "行動クローニングの右旋回です"),
    "PickAndCarry_walk.mp4": ("歩行", "止まる・歩く・曲がるを、腕を動かしながら続けます"),
    "PickAndCarry_reach.mp4": ("到達", "歩行の重みから続け、空中の点に手先を届かせます"),
    "PickAndCarry_ppo.mp4": ("箱運び", "保存した重みで、拾って目標へ置きます"),
    "PickAndCarry_z1.mp4": ("箱運び（Z1）", "Z1 の重みで、拾って目標へ置きます"),
    "PickAndCarry_haul.mp4": ("歩きながら持つ", "到達の重みから、箱を持ったまま歩きます"),
    "PickAndCarry_catch_static.mp4": ("キャッチ（静止）", "手の前で止まっているボールを掴みます"),
    "PickAndCarry_catch_slow.mp4": ("キャッチ（ゆっくり）", "ゆっくり転がるボールを掴みます"),
    "PickAndCarry_catch_random.mp4": ("キャッチ（ばらす）", "速さと方向がばらつくボールを掴みます"),
    "PickAndCarry_catch_fly.mp4": ("キャッチ（飛ぶ）", "転がってから飛んでくるボールを掴みます"),
    "PickAndCarry_catch_noisy.mp4": ("キャッチ（ノイズ）", "見ている位置にノイズがあるボールを掴みます"),
    "PickAndCarry_catch.mp4": ("転がるボール", "前の段階から続けた、キャッチの方策です"),
    "PickAndCarry_compare.mp4": ("学習の途中経過", "早い時点と遅い時点を並べます"),
    "PickAndCarry_z1_compare.mp4": ("学習の途中経過（Z1）", "早い時点と遅い時点を並べます"),
}
OUTPUT_NAME = "PickAndCarry_presentation.mp4"
# Allow a little headroom so full narrations are not rejected after padding.
MAX_TOTAL_SECONDS = 210.0
# Per-recording cap so demos/playbacks do not dominate the reel.
# Skill clips (forward/turn) are shorter; stage plays can use the full cap.
CLIP_MAX_SECONDS = 10.0
SKILL_CLIP_MAX_SECONDS = 6.0
SKILL_CLIP_NAMES = frozenset(
    {
        "trot_teacher_forward.mp4",
        "trot_teacher_turn_left.mp4",
        "trot_teacher_turn_right.mp4",
        "bc_trot_forward.mp4",
        "bc_trot_turn_left.mp4",
        "bc_trot_turn_right.mp4",
    }
)
OPENING_PAGES = (1, 2, 3, 5, 6)
CLOSING_PAGES = (8, 9)  # P8 curves → P9 conclusion
# Kept for CLI compatibility with --opening-slides / --closing-slides.
OPENING_SLIDES = len(OPENING_PAGES)
CLOSING_SLIDES = len(CLOSING_PAGES)
# Quiet beat between slides / before the next section (seconds).
TRANSITION_SECONDS = 2.5
# Spoken lines use readings that Microsoft Haruka will not mis-parse
# (e.g. 残差 → ざんさ, not のこりさ). Visual slide text stays in kanji.
OPENING_NARRATION = {
    1: "最終課題の目的です。ゴーツーにゼットワンを載せ、不整地を歩いて箱を所定位置まで運ぶほうさくを学習します。",
    2: (
        "ロボットと問題設定です。よんそくじゅうにかんせつとろくじくアームで、"
        "平地から不整地を渡り、箱を拾い、目標へ置きます。"
    ),
    3: "せいぎょほうしんです。あしはシーピージー、腕はかいせきアイケーとざんさです。",
    5: "かんそくとせいぎょりょうです。かんそくははちじゅういち、こうどうはにじゅうにです。",
    6: (
        "学習の流れです。教師シーピージーをビーシーでまね、"
        "ピーピーオーで歩行、とうたつ、はこはこびとだんかいを上げます。"
    ),
}
CLOSING_NARRATION = {
    8: (
        "ロスと学習曲線です。ほうさくロス、かちロス、エントロピーで更新します。"
        "ほうしゅうやはじりつが上がり、距離とロスが下がるほど良いです。"
    ),
    9: (
        "結論です。歩行の安定化や段階のつなぎに苦労しました。"
        "キャッチなど、より高度なタスクまで進めたい意図はありましたが、"
        "計算リソースが足りず、十分な反復を回せませんでした。"
        "そのため解析アイケーなどで高速化しつつ、段階を絞って提出しています。"
    ),
}
LAUNCHER_PPTX = LAUNCHER / "PickAndCarry_学習条件_改版.pptx"
VIDEO_NARRATION = {
    "PickAndCarry_smoke.mp4": "起動確認です。ロボットが足場に立つかを見ます。",
    "PickAndCarry_z1_smoke.mp4": "ゼットワンでの起動確認です。",
    "trot_teacher_forward.mp4": "教師データのぜんしんです。オープンループのシーピージーが前へ進みます。",
    "trot_teacher_turn_left.mp4": "教師データの左せんかいです。",
    "trot_teacher_turn_right.mp4": "教師データの右せんかいです。",
    "bc_trot_forward.mp4": "ビーシーのぜんしんです。教師をもほうしたほうさくです。",
    "bc_trot_turn_left.mp4": "ビーシーの左せんかいです。",
    "bc_trot_turn_right.mp4": "ビーシーの右せんかいです。",
    "PickAndCarry_walk.mp4": "カリキュラムの歩行です。速度しれいに合わせて歩きます。",
    "PickAndCarry_reach.mp4": "とうたつです。歩行のおもみから続け、空中の点へ、てさきを届けます。",
    "PickAndCarry_ppo.mp4": "はこはこびです。箱を拾い、目標へ置きます。",
    "PickAndCarry_z1.mp4": "はこはこびです。ゼットワンで拾って運び、目標へ置きます。",
    "PickAndCarry_haul.mp4": "持ち歩きです。箱を持ったまま歩きます。",
    "PickAndCarry_catch_static.mp4": "キャッチの静止です。",
    "PickAndCarry_catch_slow.mp4": "キャッチのゆっくりです。",
    "PickAndCarry_catch_random.mp4": "キャッチのばらすです。",
    "PickAndCarry_catch_fly.mp4": "キャッチの飛ぶです。",
    "PickAndCarry_catch_noisy.mp4": "キャッチのノイズです。",
    "PickAndCarry_catch.mp4": "キャッチです。",
    "PickAndCarry_compare.mp4": "学習の途中経過です。",
    "PickAndCarry_z1_compare.mp4": "学習の途中経過です。",
}

# Fallback for any narration that still contains technical kanji.
_SPEECH_REPLACEMENTS = (
    ("残差", "ざんさ"),
    ("開ループ", "オープンループ"),
    ("対角トロット", "たいかくトロット"),
    ("制御方針", "せいぎょほうしん"),
    ("制御量", "せいぎょりょう"),
    ("方策", "ほうさく"),
    ("観測", "かんそく"),
    ("模倣", "もほう"),
    ("到達", "とうたつ"),
    ("旋回", "せんかい"),
    ("前進", "ぜんしん"),
    ("箱運び", "はこはこび"),
    ("手先", "てさき"),
    ("重み", "おもみ"),
    ("指令", "しれい"),
    ("解析ヤコビアン", "かいせきヤコビアン"),
    ("解析", "かいせき"),
    ("数値", "すうち"),
    ("次元", "じげん"),
    ("関節", "かんせつ"),
    ("段階", "だんかい"),
    ("四足", "よんそく"),
    ("六軸", "ろくじく"),
    ("十二", "じゅうに"),
    ("十五", "じゅうご"),
    ("八十一", "はちじゅういち"),
    ("二十二", "にじゅうに"),
)


def for_speech(text: str) -> str:
    """Rewrite kanji that the Japanese TTS voice often misreads."""
    out = text
    for src, dst in _SPEECH_REPLACEMENTS:
        out = out.replace(src, dst)
    return out


# Log/play folders often keep algorithm prefixes (ppo_/grpo_) in the basename.
NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "PickAndCarry_walk.mp4": ("PickAndCarry_ppo_walk.mp4", "PickAndCarry_grpo_walk.mp4"),
    "PickAndCarry_reach.mp4": ("PickAndCarry_ppo_reach.mp4", "PickAndCarry_grpo_reach.mp4"),
    "PickAndCarry_haul.mp4": ("PickAndCarry_ppo_haul.mp4",),
    "PickAndCarry_z1.mp4": ("PickAndCarry_ppo_z1.mp4", "PickAndCarry_z1.mp4"),
    "PickAndCarry_catch.mp4": ("PickAndCarry_ppo_catch.mp4",),
}


def _newest_named(folder: Path, name: str) -> Path | None:
    candidates = (name,) + NAME_ALIASES.get(name, ())
    matches: list[Path] = []
    for candidate in candidates:
        direct = folder / candidate
        if direct.is_file():
            matches.append(direct)
        matches.extend(path for path in folder.rglob(candidate) if path.is_file())
    if not matches:
        return None
    # Prefer exact preferred basename when several aliases exist.
    exact = [path for path in matches if path.name == name]
    pool = exact or matches
    return max(pool, key=lambda path: path.stat().st_mtime)


def presentation_videos(folder: Path) -> list[Path]:
    """Collect clips by preferred basename (flat or under videos/<task>/<run>/)."""
    chosen: list[Path] = []
    seen_paths: set[Path] = set()
    for name in PREFERRED:
        path = _newest_named(folder, name)
        if path is None or path in seen_paths:
            continue
        chosen.append(path)
        seen_paths.add(path)
    # Do not dump every skill clip; only preferred stage reels belong in the submission.
    return chosen


def canonical_clip_name(path: Path) -> str:
    """Map ``ppo_`` / ``grpo_`` play filenames back to the preferred title key."""
    name = path.name
    if name in TITLES or name in VIDEO_NARRATION:
        return name
    for preferred, aliases in NAME_ALIASES.items():
        if name in aliases:
            return preferred
    return name


STAGE_CLIPS = (
    ("trot_teacher_forward.mp4", "教師・前進"),
    ("trot_teacher_turn_left.mp4", "教師・左旋回"),
    ("trot_teacher_turn_right.mp4", "教師・右旋回"),
    ("bc_trot_forward.mp4", "BC・前進"),
    ("bc_trot_turn_left.mp4", "BC・左旋回"),
    ("bc_trot_turn_right.mp4", "BC・右旋回"),
    ("PickAndCarry_walk.mp4", "歩行"),
    ("PickAndCarry_reach.mp4", "到達"),
    ("PickAndCarry_haul.mp4", "歩きながら持つ"),
    ("PickAndCarry_z1.mp4", "箱運び"),
    ("PickAndCarry_catch_static.mp4", "キャッチ（静止）"),
    ("PickAndCarry_catch_slow.mp4", "キャッチ（ゆっくり）"),
    ("PickAndCarry_catch_random.mp4", "キャッチ（ばらす）"),
    ("PickAndCarry_catch_fly.mp4", "キャッチ（飛ぶ）"),
    ("PickAndCarry_catch_noisy.mp4", "キャッチ（ノイズ）"),
)


def report_stage_clips(folder: Path) -> None:
    """Say which curriculum recordings the final video can already show."""
    print("提出動画に入る段階:", flush=True)
    for name, label in STAGE_CLIPS:
        path = _newest_named(folder, name)
        if path is not None:
            rel = path.relative_to(folder) if path.is_relative_to(folder) else path
            print(f"  あり  {label}  {rel}", flush=True)
        else:
            hint = "メニュー 1 の模倣学習" if "trot" in name or name.startswith("bc_") else "メニュー 2 でその段階を学習"
            print(f"  まだない  {label}  {hint}すると videos/<タスク>/<学習回>/{name} が残ります", flush=True)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "YuGothB.ttc" if bold else "YuGothM.ttc"
    return ImageFont.truetype(str(Path(os.environ["WINDIR"]) / "Fonts" / filename), size)


def title_card(title: str, subtitle: str, path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), INK)
    draw = ImageDraw.Draw(image)
    draw.text((80, 220), "Pick-and-Carry", font=_font(28), fill=(180, 190, 186))
    draw.text((80, 280), title, font=_font(64, bold=True), fill=WHITE)
    draw.text((80, 380), subtitle, font=_font(32), fill=(210, 214, 208))
    image.save(path)


def export_slides(pptx: Path, folder: Path) -> list[Path]:
    """Ask PowerPoint for one PNG per slide. An already-open deck is left open."""
    folder.mkdir(parents=True, exist_ok=True)
    script = r"""
$ErrorActionPreference = 'Stop'
$pptx = $env:PRES_PPTX
$dest = $env:PRES_SLIDES
$already = Get-Process POWERPNT -ErrorAction SilentlyContinue
$ppt = New-Object -ComObject PowerPoint.Application
$openedHere = $false
$pres = $null
foreach ($item in @($ppt.Presentations)) {
    if ($item.FullName -eq $pptx) { $pres = $item; break }
}
if ($null -eq $pres) {
    $pres = $ppt.Presentations.Open($pptx, $true, $false, $false)
    $openedHere = $true
}
try {
    for ($i = 1; $i -le $pres.Slides.Count; $i++) {
        $png = Join-Path $dest ('slide_{0:d2}.png' -f $i)
        $pres.Slides.Item($i).Export($png, 'PNG', 1920, 1080) | Out-Null
    }
} finally {
    if ($openedHere) { $pres.Close() }
    if (-not $already) { $ppt.Quit() }
}
"""
    env = os.environ.copy()
    env["PRES_PPTX"] = str(pptx)
    env["PRES_SLIDES"] = str(folder)
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, env=env)
    slides = sorted(folder.glob("slide_*.png"))
    if not slides:
        raise SystemExit(f"PowerPoint がスライドを書き出せませんでした: {pptx}")
    return slides


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def still_clip(ffmpeg: str, image: Path, seconds: float, output: Path) -> None:
    _run(
        [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-i",
            str(image),
            "-t",
            f"{seconds:.2f}",
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0xF6F3EC,format=yuv420p",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output),
        ]
    )


def transition_gap(ffmpeg: str, seconds: float, output: Path) -> None:
    """Silent paper-colored pause between slides / sections (with silent AAC)."""
    _run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0xF6F3EC:s=1280x720:r={FPS}:d={float(seconds):.2f}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(output),
        ]
    )


def movie_clip(ffmpeg: str, source: Path, output: Path, max_seconds: float | None = None) -> None:
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x111111,fps=25,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-an",
    ]
    if max_seconds is not None and max_seconds > 0:
        command.extend(["-t", f"{float(max_seconds):.2f}"])
    command.append(str(output))
    _run(command)


def branding_clip(ffmpeg: str, source: Path, output: Path) -> None:
    """Normalize OP/ED to 1280x720@25 with silent stereo AAC (matches body concat)."""
    if not source.is_file():
        raise SystemExit(f"OP/ED がありません: {source}")
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x000000,fps=25,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(output),
        ]
    )


def normalize_body_for_bookends(ffmpeg: str, body: Path, output: Path) -> None:
    """Scale body to 1280x720@25 with stereo AAC (keep source audio, else silence)."""
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(body)], capture_output=True, text=True)
    has_audio = "Audio:" in probe.stderr
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(body),
    ]
    if not has_audio:
        command.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"])
    command.extend(
        [
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x111111,fps=25,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0" if has_audio else "1:a:0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ac",
            "2",
            "-ar",
            "44100",
        ]
    )
    if not has_audio:
        command.append("-shortest")
    command.append(str(output))
    _run(command)


def wrap_with_bookends(ffmpeg: str, body: Path, output: Path, work: Path) -> None:
    """Write ``OP + body + ED`` to ``output``."""
    op_clip = work / "op_fit.mp4"
    ed_clip = work / "ed_fit.mp4"
    body_fit = work / "body_fit.mp4"
    branding_clip(ffmpeg, OP_PATH, op_clip)
    branding_clip(ffmpeg, ED_PATH, ed_clip)
    normalize_body_for_bookends(ffmpeg, body, body_fit)
    concat_clips(ffmpeg, [op_clip, body_fit, ed_clip], output, audio=True)


def media_duration(ffmpeg: str, path: Path) -> float:
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if match is None:
        raise RuntimeError(f"長さを読めません: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def speak(text: str, wav: Path) -> float:
    """Write narration with the installed Japanese voice and return its length."""
    env = os.environ.copy()
    env["PRES_TEXT"] = for_speech(text)
    env["PRES_WAV"] = str(wav)
    script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('Microsoft Haruka Desktop')
$synth.Rate = -1
$synth.SetOutputToWaveFile($env:PRES_WAV)
$synth.Speak($env:PRES_TEXT)
$synth.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, env=env)
    return media_duration(imageio_ffmpeg.get_ffmpeg_exe(), wav)


def mux_voice(ffmpeg: str, video: Path, wav: Path, output: Path) -> None:
    """Mux narration onto video without cutting the speech short.

    If the wav is longer than the picture, the last video frame is frozen until
    the voice finishes. If the picture is longer, audio is padded with silence.
    """
    v_sec = media_duration(ffmpeg, video)
    a_sec = media_duration(ffmpeg, wav)
    out_sec = max(v_sec, a_sec) + 0.15
    pad_v = max(0.0, out_sec - v_sec)
    if pad_v > 0.05:
        # Re-encode so tpad can freeze the last frame for the remaining speech.
        _run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video),
                "-i",
                str(wav),
                "-filter_complex",
                f"[0:v]tpad=stop_mode=clone:stop_duration={pad_v:.2f},"
                f"fps={FPS},format=yuv420p[v];"
                f"[1:a]aresample=44100,aformat=channel_layouts=stereo,"
                f"apad=whole_dur={out_sec:.2f}[a]",
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-t",
                f"{out_sec:.2f}",
                str(output),
            ]
        )
        return
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-i",
            str(wav),
            "-filter_complex",
            f"[1:a]aresample=44100,aformat=channel_layouts=stereo,"
            f"apad=whole_dur={out_sec:.2f}[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-t",
            f"{out_sec:.2f}",
            str(output),
        ]
    )


def concat_clips(ffmpeg: str, clips: list[Path], output: Path, audio: bool = False) -> None:
    listing = clips[0].with_name("_concat.txt")
    lines = []
    for clip in clips:
        escaped = str(clip.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
    ]
    if audio:
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")
    command.append(str(output))
    _run(command)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a presentation mp4 from the slides and recordings.")
    parser.add_argument("--pptx", type=Path, default=LAUNCHER_PPTX)
    parser.add_argument("--videos", type=Path, default=LAUNCHER)
    parser.add_argument("--output", type=Path, default=LAUNCHER / OUTPUT_NAME)
    parser.add_argument("--slide-seconds", type=float, default=6.0, help="Minimum seconds per slide.")
    parser.add_argument(
        "--slide-max-seconds",
        type=float,
        default=10.0,
        help="Preferred max when speech is short; never shorter than the narration.",
    )
    parser.add_argument("--card-seconds", type=float, default=1.5)
    parser.add_argument("--clip-seconds", type=float, default=CLIP_MAX_SECONDS, help="Max seconds per recording.")
    parser.add_argument("--max-seconds", type=float, default=MAX_TOTAL_SECONDS, help="Abort if longer than this.")
    parser.add_argument("--opening-slides", type=int, default=OPENING_SLIDES)
    parser.add_argument("--closing-slides", type=int, default=CLOSING_SLIDES)
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=TRANSITION_SECONDS,
        help="Quiet seconds between slides / before the next section.",
    )
    parser.add_argument(
        "--wrap-body",
        type=Path,
        default=None,
        help="Skip slide assembly; only prepend OP and append ED around this mp4.",
    )
    parser.add_argument(
        "--no-bookends",
        action="store_true",
        help="Do not attach PAI応用1_最終課題_OP/ED.mp4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.wrap_body is not None:
        body = args.wrap_body
        if not body.is_file():
            raise SystemExit(f"本編がありません: {body}")
        if args.no_bookends:
            raise SystemExit("--wrap-body と --no-bookends は同時に使えません")
        with tempfile.TemporaryDirectory(prefix="pac_wrap_") as tmp:
            wrap_with_bookends(ffmpeg, body, args.output, Path(tmp))
        print(f"提出動画（OP+本編+ED）: {args.output.resolve()}")
        return

    if not args.pptx.is_file():
        raise SystemExit(f"スライドがありません: {args.pptx}")
    videos = presentation_videos(args.videos)
    report_stage_clips(args.videos)
    if not videos:
        raise SystemExit(f"つなぐ mp4 がありません: {args.videos}")
    print(f"スライド: {args.pptx}")
    print("録画:")
    for video in videos:
        print(f"  {video.name}")
    with tempfile.TemporaryDirectory(prefix="pac_pres_") as tmp:
        work = Path(tmp)
        slides = export_slides(args.pptx, work / "slides")
        clips: list[Path] = []

        gap_n = 0

        def append_gap(reason: str) -> None:
            nonlocal gap_n
            if args.transition_seconds <= 0:
                return
            gap_n += 1
            path = work / f"gap_{gap_n:02d}.mp4"
            transition_gap(ffmpeg, args.transition_seconds, path)
            clips.append(path)
            print(f"間 {args.transition_seconds:.1f} 秒（{reason}）", flush=True)

        def append_slide(label: str, slide: Path, text: str, *, gap_after: bool = True) -> None:
            wav = work / f"{label}.wav"
            spoken = speak(text, wav)
            # Always cover the full spoken line; slide_max only shortens quiet slides.
            seconds = max(spoken + 0.8, min(args.slide_max_seconds, args.slide_seconds))
            silent = work / f"{label}_silent.mp4"
            clip = work / f"{label}.mp4"
            still_clip(ffmpeg, slide, seconds, silent)
            mux_voice(ffmpeg, silent, wav, clip)
            out_sec = media_duration(ffmpeg, clip)
            clips.append(clip)
            print(
                f"{label} を {out_sec:.0f} 秒（読み上げ {spoken:.1f} 秒）で入れました。",
                flush=True,
            )
            if gap_after:
                append_gap(label)

        if not args.no_bookends:
            op_clip = work / "op_fit.mp4"
            branding_clip(ffmpeg, OP_PATH, op_clip)
            clips.append(op_clip)
            print(f"OP を先頭に追加: {OP_PATH.name}", flush=True)
            append_gap("OPのあと")

        opening_pages = [p for p in OPENING_PAGES if 1 <= p <= len(slides)]
        closing_pages = [p for p in CLOSING_PAGES if 1 <= p <= len(slides)]
        print(
            f"構成: P{','.join(map(str, opening_pages))} → 学習動画 → P{','.join(map(str, closing_pages))}",
            flush=True,
        )
        for page in opening_pages:
            text = OPENING_NARRATION.get(page, "続きのスライドです。")
            append_slide(f"slide_{page:02d}", slides[page - 1], text)
        for video in videos:
            key = canonical_clip_name(video)
            title, subtitle = TITLES.get(key, (video.stem, "録画"))
            text = VIDEO_NARRATION.get(key, f"{title}。{subtitle}")
            wav = work / f"{video.stem}.wav"
            spoken = speak(text, wav)
            card = work / f"{video.stem}_card.png"
            title_card(title, subtitle, card)
            movie = work / f"{video.stem}_fit.mp4"
            clip_cap = (
                min(args.clip_seconds, SKILL_CLIP_MAX_SECONDS)
                if canonical_clip_name(video) in SKILL_CLIP_NAMES
                else args.clip_seconds
            )
            movie_clip(ffmpeg, video, movie, max_seconds=clip_cap)
            movie_seconds = media_duration(ffmpeg, movie)
            # Title card absorbs leftover speech time so mux_voice never clips the wav.
            card_seconds = max(args.card_seconds, spoken - movie_seconds + 0.8)
            if spoken <= movie_seconds:
                card_seconds = min(args.card_seconds, 1.2)
            card_clip = work / f"{video.stem}_card.mp4"
            still_clip(ffmpeg, card, card_seconds, card_clip)
            silent = work / f"{video.stem}_silent.mp4"
            clip = work / f"{video.stem}.mp4"
            concat_clips(ffmpeg, [card_clip, movie], silent)
            mux_voice(ffmpeg, silent, wav, clip)
            clips.append(clip)
            print(
                f"録画を追加: {video.name}（映像 {movie_seconds:.0f} 秒、"
                f"カード {card_seconds:.1f} 秒、読み上げ {spoken:.1f} 秒）",
                flush=True,
            )
        append_gap("学習動画のあと")
        for i, page in enumerate(closing_pages):
            text = CLOSING_NARRATION.get(page, "まとめのスライドです。")
            # Keep a beat between closing slides; skip after the last before ED.
            append_slide(
                f"slide_{page:02d}",
                slides[page - 1],
                text,
                gap_after=(i + 1 < len(closing_pages)),
            )

        if not args.no_bookends:
            append_gap("EDのまえ")
            ed_clip = work / "ed_fit.mp4"
            branding_clip(ffmpeg, ED_PATH, ed_clip)
            clips.append(ed_clip)
            print(f"ED を末尾に追加: {ED_PATH.name}", flush=True)

        concat_clips(ffmpeg, clips, args.output, audio=True)
    duration = media_duration(ffmpeg, args.output)
    print(f"プレゼン動画: {args.output.resolve()}（{duration:.1f} 秒）")
    if duration > args.max_seconds + 0.5:
        raise SystemExit(
            f"提出上限 {args.max_seconds:.0f} 秒を超えています（{duration:.1f} 秒）。"
            " --clip-seconds や --opening-slides を下げて再実行してください。"
        )


if __name__ == "__main__":
    main()
