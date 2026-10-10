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
LAUNCHER = Path.home() / "Desktop" / "起動スクリプト"
WIDTH = 1280
HEIGHT = 720
FPS = 25
INK = (28, 40, 44)
WHITE = (255, 255, 255)

# Narrative order. Anything else in the folder is appended after these.
PREFERRED = (
    "PickAndCarry_smoke.mp4",
    "PickAndCarry_ppo.mp4",
    "PickAndCarry_compare.mp4",
    "PickAndCarry_supplies_90s.mp4",
)
TITLES = {
    "PickAndCarry_smoke.mp4": ("起動確認", "ロボットが足場に立つかを見ます"),
    "PickAndCarry_ppo.mp4": ("学習した方策", "保存した重みでコースを再生します"),
    "PickAndCarry_compare.mp4": ("学習の途中経過", "早い時点と遅い時点を並べます"),
    "PickAndCarry_supplies_90s.mp4": ("複数の物資を運ぶ", "台本の歩行で、箱を順に拾って置きます"),
}
OUTPUT_NAME = "PickAndCarry_presentation.mp4"
SLIDE_NARRATION = (
    "学習条件です。ユニットリーのゴーツーの背中に、フランカの腕を載せています。"
    "学習はピーピーオーで、観測は68次元、制御は20次元です。この寸法のまま、続きから学習できます。"
    "脚と腕と指は位置で動かし、段階は、箱へ歩く、拾う、運ぶ、置く、の4つです。",
    "問題設定です。平らな場所から出発し、不整地を渡って箱を拾います。"
    "もう一度不整地を渡り、目標に置きます。箱と目標の下は平らです。"
    "茶色の段は、段差を選んだときだけで、箱の前に1段、置く前に2段です。"
    "1試行は20秒です。目標の近くに低く置くと成功で、大きく傾くと転倒です。",
    "報酬の説明です。秒速0.4メートルで前に進むと点が増えます。"
    "箱に手を近づける、つかむ、運ぶ、目標に置く、でも点が増えます。"
    "置けた瞬間は10点、転ぶと5点の減点です。横滑りと、大きすぎる操作も減点です。",
    "損失と学習曲線です。方策のロスは、良かった行動を少しずつ増やす量で、"
    "一度の変化はクリップ0.2までに制限します。価値のロスは、これからの報酬の予測誤差で、係数は0.5です。"
    "エントロピーは0.005で、探索を残します。"
    "曲線では、報酬と把持率は上がるほど良く、手先と箱の距離は下がるほど良いです。"
    "方策のロスは、ゼロの近くに落ち着きます。",
)
# Slides before this count play first. Later slides, including reward and curves, play after the recordings.
OPENING_SLIDES = 2
VIDEO_NARRATION = {
    "PickAndCarry_smoke.mp4": "起動確認です。ロボットが足場に立つかを見ます。",
    "PickAndCarry_ppo.mp4": "学習した方策の再生です。保存した重みで、不整地を渡って箱を拾い、目標へ運ぶ様子です。",
    "PickAndCarry_compare.mp4": "学習の途中経過です。同じ配置で、早い時点と遅い時点を同時に再生しています。",
    "PickAndCarry_supplies_90s.mp4": "複数の物資を運ぶ実演です。台本の歩行で、箱を順に拾って置きます。",
}


def presentation_videos(folder: Path) -> list[Path]:
    files = [path for path in folder.glob("*.mp4") if path.name != OUTPUT_NAME]
    rank = {name: index for index, name in enumerate(PREFERRED)}
    return sorted(files, key=lambda path: (rank.get(path.name, len(PREFERRED)), path.name.lower()))


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


def movie_clip(ffmpeg: str, source: Path, output: Path) -> None:
    _run(
        [
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
            str(output),
        ]
    )


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
    env["PRES_TEXT"] = text
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
    seconds = media_duration(ffmpeg, video)
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-i",
            str(wav),
            "-filter_complex",
            f"[1:a]aresample=44100,aformat=channel_layouts=stereo,apad=whole_dur={seconds + 0.2:.2f}[a]",
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
            f"{seconds:.2f}",
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
    parser.add_argument("--pptx", type=Path, default=LAUNCHER / "PickAndCarry_学習条件.pptx")
    parser.add_argument("--videos", type=Path, default=LAUNCHER)
    parser.add_argument("--output", type=Path, default=LAUNCHER / OUTPUT_NAME)
    parser.add_argument("--slide-seconds", type=float, default=12.0)
    parser.add_argument("--card-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pptx.is_file():
        raise SystemExit(f"スライドがありません: {args.pptx}")
    videos = presentation_videos(args.videos)
    if not videos:
        raise SystemExit(f"つなぐ mp4 がありません: {args.videos}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"スライド: {args.pptx}")
    print("録画:")
    for video in videos:
        print(f"  {video.name}")
    with tempfile.TemporaryDirectory(prefix="pac_pres_") as tmp:
        work = Path(tmp)
        slides = export_slides(args.pptx, work / "slides")
        clips: list[Path] = []

        def append_slide(index: int, slide: Path) -> None:
            text = SLIDE_NARRATION[index - 1] if index - 1 < len(SLIDE_NARRATION) else "続きのスライドです。"
            wav = work / f"slide_{index:02d}.wav"
            spoken = speak(text, wav)
            seconds = max(args.slide_seconds, spoken + 0.6)
            silent = work / f"slide_{index:02d}_silent.mp4"
            clip = work / f"slide_{index:02d}.mp4"
            still_clip(ffmpeg, slide, seconds, silent)
            mux_voice(ffmpeg, silent, wav, clip)
            clips.append(clip)
            print(f"スライド {index}/{len(slides)} を {seconds:.0f} 秒、説明つきで入れました。")

        for index, slide in enumerate(slides[:OPENING_SLIDES], start=1):
            append_slide(index, slide)
        for video in videos:
            title, subtitle = TITLES.get(video.name, (video.stem, "録画"))
            text = VIDEO_NARRATION.get(video.name, f"{title}。{subtitle}")
            wav = work / f"{video.stem}.wav"
            spoken = speak(text, wav)
            card = work / f"{video.stem}_card.png"
            title_card(title, subtitle, card)
            movie = work / f"{video.stem}_fit.mp4"
            movie_clip(ffmpeg, video, movie)
            movie_seconds = media_duration(ffmpeg, movie)
            card_seconds = max(args.card_seconds, spoken - movie_seconds + 0.6)
            card_clip = work / f"{video.stem}_card.mp4"
            still_clip(ffmpeg, card, card_seconds, card_clip)
            silent = work / f"{video.stem}_silent.mp4"
            clip = work / f"{video.stem}.mp4"
            concat_clips(ffmpeg, [card_clip, movie], silent)
            mux_voice(ffmpeg, silent, wav, clip)
            clips.append(clip)
            print(f"録画を追加: {video.name}（説明 {spoken:.0f} 秒）")
        for offset, slide in enumerate(slides[OPENING_SLIDES:]):
            append_slide(OPENING_SLIDES + offset + 1, slide)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        concat_clips(ffmpeg, clips, args.output, audio=True)
    print(f"プレゼン動画: {args.output.resolve()}")


if __name__ == "__main__":
    main()
