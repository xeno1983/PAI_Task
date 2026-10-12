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
    "PickAndCarry_z1_smoke.mp4",
    "PickAndCarry_walk.mp4",
    "PickAndCarry_rough.mp4",
    "PickAndCarry_ppo.mp4",
    "PickAndCarry_z1.mp4",
    "PickAndCarry_haul.mp4",
    "PickAndCarry_catch.mp4",
    "PickAndCarry_compare.mp4",
    "PickAndCarry_z1_compare.mp4",
    "PickAndCarry_supplies_90s.mp4",
)
TITLES = {
    "PickAndCarry_smoke.mp4": ("起動確認", "ロボットが足場に立つかを見ます"),
    "PickAndCarry_z1_smoke.mp4": ("起動確認（Z1）", "軽い腕でも足場に立つかを見ます"),
    "PickAndCarry_walk.mp4": ("平地の歩行", "速度の指令に合わせて歩く方策です"),
    "PickAndCarry_rough.mp4": ("不整地の歩行", "平地の重みから続けた歩行です"),
    "PickAndCarry_ppo.mp4": ("箱運び", "保存した重みで、拾って目標へ置きます"),
    "PickAndCarry_z1.mp4": ("箱運び（Z1）", "Z1 の重みで、拾って目標へ置きます"),
    "PickAndCarry_haul.mp4": ("歩きながら持つ", "覚えた脚で歩き、腕は箱を持ったままです"),
    "PickAndCarry_catch.mp4": ("転がるボール", "前の段階から続けた、キャッチの方策です"),
    "PickAndCarry_compare.mp4": ("学習の途中経過", "早い時点と遅い時点を並べます"),
    "PickAndCarry_z1_compare.mp4": ("学習の途中経過（Z1）", "早い時点と遅い時点を並べます"),
    "PickAndCarry_supplies_90s.mp4": ("複数の物資を運ぶ", "台本の歩行で、箱を順に拾って置きます"),
}
OUTPUT_NAME = "PickAndCarry_presentation.mp4"
SLIDE_NARRATION = (
    "学習条件です。ユニットリーのゴーツーに、フランカのパンダ、またはゼットワンを載せます。"
    "学習はピーピーオーです。箱運びは、パンダなら観測68、制御20。ゼットワンなら観測66、制御19です。"
    "形が違う重みは続きに使いません。パンダは ppo walk、ゼットワンは ppo z1 から再開します。"
    "脚と腕と指は位置で動かし、箱運びの段階は、歩く、拾う、運ぶ、置く、の4つです。",
    "問題設定です。平らな場所から出発し、不整地を渡って箱を拾います。"
    "もう一度不整地を渡り、目標に置きます。箱と目標の下は平らです。"
    "茶色の段は、段差を選んだときだけで、箱の前に1段、置く前に2段です。"
    "1試行は20秒です。目標の近くに低く置くと成功で、大きく傾くと転倒です。",
    "段階的な学習です。最初は平地で、前後、左右、旋回の指令に合わせて歩きます。"
    "次の不整地は、その重みから続けます。箱運びは別のネットワークで、今までの保存を使います。"
    "歩きながら持つ段階では、覚えた脚を固定し、腕と進む速さを学びます。箱は手に固定しています。"
    "最後は転がるボールです。静止、ゆっくり、ばらす、ノイズ、の順で、近いレベルの重みから続けます。"
    "1秒保持すると成功です。次の段階へ進むかは、録画を見て決めます。人の点数で報酬は置き換えていません。",
    "学習の工夫です。歩行と把持を一度に覚えさせず、段階に分けています。"
    "形が同じ段階だけ、前の重みを引き継ぎます。形が違うときは、覚えた歩行を脚として固定し、腕と進む速さだけを新しく学びます。"
    "報酬は、成功した瞬間だけではなく、毎ステップの接近と、速度の一致でも点を出します。"
    "不整地で成績が低いあいだは平地に戻し、キャッチで成功が少ないあいだは易しいボールに戻します。"
    "8回続いて回復したら、選んだ課題まで上げます。メニューからは、上の段階の重みで易しい課題を学び直すこともできます。",
    "報酬の説明です。箱運びでは、秒速0.4メートルで前に進むと点が増えます。"
    "箱に手を近づける、つかむ、運ぶ、目標に置く、でも点が増えます。"
    "置けた瞬間は10点、転ぶと5点の減点です。横滑りと、大きすぎる操作も減点です。"
    "歩行の段階では、この表の代わりに、指令した速度への一致が点になります。"
    "キャッチでは、手をボールへ近づけること、速度を揃えること、指のあいだで保持することが点になります。",
    "損失と学習曲線です。方策のロスは、良かった行動を少しずつ増やす量で、"
    "一度の変化はクリップ0.2までに制限します。価値のロスは、これからの報酬の予測誤差で、係数は0.5です。"
    "エントロピーは0.005で、探索を残します。"
    "箱運びの曲線では、報酬と把持率は上がるほど良く、手先と箱の距離は下がるほど良いです。"
    "歩行の記録はトラッキングで、1に近いほど指令どおりです。方策のロスは、ゼロの近くに落ち着きます。",
)
# Conditions, the course, the curriculum, and the training techniques play first.
OPENING_SLIDES = 4
VIDEO_NARRATION = {
    "PickAndCarry_smoke.mp4": "起動確認です。ロボットが足場に立つかを見ます。",
    "PickAndCarry_z1_smoke.mp4": "ゼットワンでの起動確認です。軽い腕でも足場に立つかを見ます。",
    "PickAndCarry_walk.mp4": "平地の歩行です。速度の指令に合わせて歩く方策を再生しています。",
    "PickAndCarry_rough.mp4": "不整地の歩行です。平地で覚えた重みから続けた方策です。",
    "PickAndCarry_ppo.mp4": "箱運びです。保存した重みで、不整地を渡って箱を拾い、目標へ運ぶ様子です。",
    "PickAndCarry_z1.mp4": "ゼットワンの箱運びです。パンダの重みは使わず、ゼットワンの保存で再生しています。",
    "PickAndCarry_haul.mp4": "歩きながら持つ段階です。脚は覚えた歩行で、腕は箱を持ったまま進みます。",
    "PickAndCarry_catch.mp4": "転がるボールを掴む段階です。前の段階の重みから続けた方策です。",
    "PickAndCarry_compare.mp4": "学習の途中経過です。同じ配置で、早い時点と遅い時点を同時に再生しています。",
    "PickAndCarry_z1_compare.mp4": "ゼットワンの途中経過です。同じ配置で、早い時点と遅い時点を同時に再生しています。",
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
        if len(slides) != len(SLIDE_NARRATION):
            print(
                f"スライドは {len(slides)} 枚、説明は {len(SLIDE_NARRATION)} 件です。",
                flush=True,
            )
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
