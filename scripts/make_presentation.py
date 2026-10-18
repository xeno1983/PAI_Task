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
LAUNCHER = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"
WIDTH = 1280
HEIGHT = 720
FPS = 25
INK = (28, 40, 44)
WHITE = (255, 255, 255)

# Narrative order. Anything else in the folder is appended after these.
PREFERRED = (
    "PickAndCarry_smoke.mp4",
    "PickAndCarry_z1_smoke.mp4",
    "trot_teacher_demo.mp4",
    "bc_trot_demo.mp4",
    "gail_trot_demo.mp4",
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
    "trot_teacher_demo.mp4": ("教師デモ（CPG）", "立つ・座る・歩く・旋回などを CPG 教師が示します"),
    "bc_trot_demo.mp4": ("BC デモ", "教師を模倣した行動クローニングの方策です"),
    "gail_trot_demo.mp4": ("GAIL デモ", "識別器報酬で強化学習した模倣方策です"),
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
SLIDE_NARRATION = (
    "学習条件です。ユニットリーのゴーツーに、ユニットリーのゼットワンを載せます。"
    "学習はピーピーオーと、行動クローニング、ゲイルです。全段階が観測81、制御17です。"
    "脚はシーピージーのパラメータを方策が出し、環境が12関節へ展開します。"
    "腕が違う古い保存や、制御19の古い重みは続きに使いません。"
    "箱運びの段階は、歩く、拾う、運ぶ、置く、の4つです。",
    "問題設定です。平らな場所から出発し、不整地を渡って箱を拾います。"
    "もう一度不整地を渡り、目標に置きます。箱と目標の下は平らです。"
    "茶色の段は、段差を選んだときだけで、箱の前に1段、置く前に2段です。"
    "1試行は20秒です。目標の近くに低く置くと成功で、大きく傾くと転倒です。",
    "段階的な学習です。最初は歩行です。平地から始めて、傾斜と段差へ広げ、腕を動かしながら歩きます。"
    "次に、空中の点へ手先を届かせる到達です。そのあと、歩きながら持つ、箱運び、ボールのキャッチの順です。"
    "同じ大きさの観測と行動なので、直前の段階の重みを初期値にします。"
    "箱運びは、歩きながら持つ段階の重みから続けます。キャッチは到達の重みから続け、静止、ゆっくり、ばらす、飛ぶ、ノイズの順に難しくします。"
    "1秒保持すると成功です。次の段階へ進むかは、録画を見て決めます。人の点数で報酬は置き換えていません。",
    "学習の工夫です。歩行と把持を一度に覚えさせず、段階に分けています。"
    "同じ観測81と行動17です。直前の段階の重みを初期値にします。"
    "報酬は、成功した瞬間だけではなく、毎ステップの接近と、速度の一致でも点を出します。"
    "歩行で成績が低いあいだは平地に戻し、キャッチで成功が少ないあいだは易しいボールに戻します。"
    "8回続いて回復したら、選んだ課題まで上げます。メニューからは、上の段階の重みで易しい課題を学び直すこともできます。",
    "報酬の説明です。箱運びでは、秒速0.4メートルで前に進むと点が増えます。"
    "箱に手を近づける、つかむ、運ぶ、目標に置く、でも点が増えます。"
    "置けた瞬間は10点、転ぶと5点の減点です。横滑りと、大きすぎる操作も減点です。"
    "歩行の段階では、この表の代わりに、指令した速度への一致が点になります。"
    "到達とキャッチでは、手をボールへ近づけること、速度を揃えること、指のあいだで保持することが点になります。",
    "損失と学習曲線です。方策のロスは、良かった行動を少しずつ増やす量で、"
    "一度の変化はクリップ0.2までに制限します。価値のロスは、これからの報酬の予測誤差で、係数は1.0です。"
    "エントロピーは0.01で、探索を残します。"
    "箱運びの曲線では、報酬と把持率は上がるほど良く、手先と箱の距離は下がるほど良いです。"
    "歩行の記録はトラッキングで、1に近いほど指令どおりです。方策のロスは、ゼロの近くに落ち着きます。",
)
# Conditions, the course, the curriculum, and the training techniques play first.
OPENING_SLIDES = 4
VIDEO_NARRATION = {
    "PickAndCarry_smoke.mp4": "起動確認です。ロボットが足場に立つかを見ます。",
    "PickAndCarry_z1_smoke.mp4": "ゼットワンでの起動確認です。軽い腕でも足場に立つかを見ます。",
    "trot_teacher_demo.mp4": "教師デモです。シーピージー教師が立つ、座る、歩く、旋回などの技能を示します。",
    "bc_trot_demo.mp4": "行動クローニングのデモです。教師のシーピージー出力を模倣した方策です。",
    "gail_trot_demo.mp4": "ゲイルのデモです。識別器の報酬で強化学習した模倣方策です。",
    "PickAndCarry_walk.mp4": "歩行です。止まる、歩く、曲がるを、腕を動かしながら続ける方策を再生しています。",
    "PickAndCarry_reach.mp4": "到達です。歩行の重みから続けて、空中の点に手先を届かせる様子です。",
    "PickAndCarry_ppo.mp4": "箱運びです。保存した重みで、不整地を渡って箱を拾い、目標へ運ぶ様子です。",
    "PickAndCarry_z1.mp4": "ゼットワンの箱運びです。歩きながら持つ段階の重みから続けた方策です。",
    "PickAndCarry_haul.mp4": "歩きながら持つ段階です。到達の重みから、箱を持ったまま歩きます。",
    "PickAndCarry_catch_static.mp4": "キャッチの静止です。手の前で止まっているボールを、1秒つかむ様子です。",
    "PickAndCarry_catch_slow.mp4": "キャッチのゆっくりです。まっすぐ転がってくるボールをつかむ様子です。",
    "PickAndCarry_catch_random.mp4": "キャッチのばらす段階です。速さと方向が毎回変わるボールをつかむ様子です。",
    "PickAndCarry_catch_fly.mp4": "キャッチの飛ぶ段階です。転がってから宙を飛んでくるボールを、手の位置に合わせてつかむ様子です。",
    "PickAndCarry_catch_noisy.mp4": "キャッチのノイズ段階です。ボールの位置を少しずらして見せたときの様子です。",
    "PickAndCarry_catch.mp4": "転がるボールを掴む段階です。前の段階の重みから続けた方策です。",
    "PickAndCarry_compare.mp4": "学習の途中経過です。同じ配置で、早い時点と遅い時点を同時に再生しています。",
    "PickAndCarry_z1_compare.mp4": "ゼットワンの途中経過です。同じ配置で、早い時点と遅い時点を同時に再生しています。",
}


def _newest_named(folder: Path, name: str) -> Path | None:
    direct = folder / name
    if direct.is_file():
        return direct
    matches = [path for path in folder.rglob(name) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def presentation_videos(folder: Path) -> list[Path]:
    """Collect clips by preferred basename (flat or under videos/<task>/<run>/)."""
    rank = {name: index for index, name in enumerate(PREFERRED)}
    chosen: list[Path] = []
    seen: set[str] = set()
    for name in PREFERRED:
        path = _newest_named(folder, name)
        if path is None:
            continue
        chosen.append(path)
        seen.add(name)
    extras = [
        path
        for path in folder.rglob("*.mp4")
        if path.name != OUTPUT_NAME and path.name not in seen and path not in chosen
    ]
    extras.sort(key=lambda path: (rank.get(path.name, len(PREFERRED)), path.stat().st_mtime, path.name.lower()))
    return chosen + extras


STAGE_CLIPS = (
    ("trot_teacher_demo.mp4", "教師デモ（CPG）"),
    ("bc_trot_demo.mp4", "BC デモ"),
    ("gail_trot_demo.mp4", "GAIL デモ"),
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
            hint = "メニュー 1 の模倣学習" if "trot" in name or name.startswith("bc_") or name.startswith("gail_") else "メニュー 2 でその段階を学習"
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
    report_stage_clips(args.videos)
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
