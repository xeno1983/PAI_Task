"""One-slide summary of the pick-and-carry training setup."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

FONT = "Yu Gothic"
INK = RGBColor(0x1C, 0x28, 0x2C)
MUTED = RGBColor(0x4A, 0x58, 0x5C)
PAPER = RGBColor(0xF6, 0xF3, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEAL = RGBColor(0x1F, 0x4E, 0x5F)
CLAY = RGBColor(0x8A, 0x45, 0x2A)
GREEN = RGBColor(0x2C, 0x64, 0x4A)


def paint(run, size, bold=False, color=INK) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    props = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        node = props.find(qn(tag))
        if node is None:
            node = props.makeelement(qn(tag), {})
            props.append(node)
        node.set("typeface", FONT)


def write(frame, lines, size=12, color=INK, bold=False, align=PP_ALIGN.LEFT, spacing=4) -> None:
    frame.word_wrap = True
    frame.auto_size = None
    text = frame.text_frame
    text.word_wrap = True
    text.auto_size = None
    for index, line in enumerate(lines):
        paragraph = text.paragraphs[0] if index == 0 else text.add_paragraph()
        paragraph.alignment = align
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(spacing)
        run = paragraph.add_run()
        run.text = line
        paint(run, size, bold=bold, color=color)


def card(slide, x, y, w, h, title, lines, accent) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.fill.background()
    shape.shadow.inherit = False
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Inches(0.38))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    write(bar, [title], size=15, bold=True, color=WHITE, align=PP_ALIGN.LEFT, spacing=0)
    bar.text_frame.margin_left = Inches(0.16)
    bar.text_frame.margin_top = Inches(0.04)
    bar.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    body = slide.shapes.add_textbox(x + Inches(0.14), y + Inches(0.46), w - Inches(0.26), h - Inches(0.56))
    write(body, lines, size=11, color=INK, spacing=3)
    body.text_frame.margin_left = Emu(0)
    body.text_frame.margin_right = Emu(0)


def _box(slide, x, y, w, h, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
    shape.shadow.inherit = False
    return shape


def _caption(slide, x, y, w, h, text, size=12, color=INK, bold=False, align=PP_ALIGN.CENTER):
    box = slide.shapes.add_textbox(x, y, w, h)
    write(box, [text], size=size, color=color, bold=bold, align=align, spacing=0)
    return box


def add_problem_slide(prs) -> None:
    """Side view of the course: problem, pads, and the rough strips."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.0), Inches(0.38))
    write(title, ["問題設定と環境"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.48), Inches(12.4), Inches(0.32))
    write(
        subtitle,
        ["四足が不整地を渡り、背中の腕で箱を拾い、もう一度不整地を渡って目標へ置く。1試行は 20 秒。"],
        size=14,
        color=MUTED,
        spacing=0,
    )

    # World x from -0.8 m to 4.8 m, drawn left to right. Height is exaggerated.
    x0, x1 = -0.8, 4.8
    left, width = 0.55, 12.25
    scale = width / (x1 - x0)
    ground_top = 3.55
    ground_h = 0.28
    step_h = 0.34

    def wx(meters: float):
        return Inches(left + (meters - x0) * scale)

    def span(a: float, b: float):
        return wx(a), Inches((b - a) * scale)

    flat = RGBColor(0xD5, 0xE6, 0xC8)
    rough = RGBColor(0xC4, 0xB4, 0x9A)
    step = RGBColor(0x8A, 0x6A, 0x45)
    _box(slide, Inches(left), Inches(ground_top), Inches(width), Inches(ground_h), RGBColor(0xE7, 0xE1, 0xD4))

    zones = (
        (-0.8, 0.55, flat, "出発"),
        (0.55, 0.85, rough, "不整地"),
        (1.05, 2.15, flat, "拾う"),
        (2.40, 3.05, rough, "不整地"),
        (3.30, 4.55, flat, "置く"),
    )
    for a, b, color, name in zones:
        zx, zw = span(a, b)
        _box(slide, zx, Inches(ground_top), zw, Inches(ground_h), color)
        label_w = max(zw, Inches(0.95))
        _caption(
            slide,
            zx - (label_w - zw) / 2,
            Inches(ground_top + ground_h + 0.06),
            label_w,
            Inches(0.28),
            name,
            size=13,
            bold=True,
        )

    for a, b in ((0.57, 0.83), (2.45, 2.68), (2.80, 3.02)):
        zx, zw = span(a, b)
        _box(slide, zx, Inches(ground_top - step_h), zw, Inches(step_h), step)

    # Robot at x = 0, cube near 1.8 m, goal near 4.0 m.
    robot_x, _ = span(0.0, 0.0)
    _box(slide, robot_x - Inches(0.22), Inches(ground_top - 0.42), Inches(0.46), Inches(0.22), RGBColor(0x5C, 0x67, 0x6C))
    _caption(slide, robot_x - Inches(0.45), Inches(ground_top - 0.72), Inches(0.9), Inches(0.26), "Go2+腕", size=11, bold=True)

    cube_x, _ = span(1.78, 1.78)
    _box(slide, cube_x - Inches(0.09), Inches(ground_top - 0.18), Inches(0.18), Inches(0.18), RGBColor(0xE0, 0x70, 0x30))
    _caption(slide, cube_x - Inches(0.4), Inches(ground_top - 0.48), Inches(0.8), Inches(0.24), "箱 4cm", size=11, color=CLAY, bold=True)

    goal_x, goal_w = span(3.75, 4.25)
    _box(slide, goal_x, Inches(ground_top - 0.08), goal_w, Inches(0.08), GREEN)
    _caption(slide, goal_x - Inches(0.15), Inches(ground_top - 0.40), goal_w + Inches(0.3), Inches(0.24), "目標", size=11, color=GREEN, bold=True)

    _caption(
        slide,
        Inches(0.55),
        Inches(1.00),
        Inches(12.2),
        Inches(0.28),
        "断面は高さを強調。茶色の段は選択時のみ（箱の前に1段、置く前に2段）。選ばなければ同じ位置が低い波。",
        size=12,
        color=MUTED,
        align=PP_ALIGN.LEFT,
    )

    phases = (
        (0.55, "0  箱へ歩く", "胴体から 0.55 m より遠い"),
        (3.70, "1  拾う", "0.55 m 以内で腕を出す"),
        (6.85, "2  運ぶ", "つかんだまま目標へ"),
        (10.00, "3  置く", "目標まで 0.45 m 以内"),
    )
    for x, head, body in phases:
        _box(slide, Inches(x), Inches(4.55), Inches(2.85), Inches(0.95), WHITE)
        _caption(slide, Inches(x + 0.1), Inches(4.62), Inches(2.65), Inches(0.32), head, size=14, bold=True, align=PP_ALIGN.LEFT)
        _caption(slide, Inches(x + 0.1), Inches(4.96), Inches(2.65), Inches(0.36), body, size=12, color=MUTED, align=PP_ALIGN.LEFT)

    notes = (
        (0.55, "地面", "箱と目標の下は平ら。波は 3.5 cm。段差は 4.5 cm または 6 cm で、位置は毎回 ±2 cm。"),
        (6.90, "成功と失敗", "目標の半径 0.28 m、箱の高さ 0.08 m 未満で成功。傾き 60° または胴体 0.18 m 未満で転倒。"),
    )
    for x, head, body in notes:
        _caption(slide, Inches(x), Inches(5.70), Inches(5.8), Inches(0.28), head, size=14, bold=True, align=PP_ALIGN.LEFT, color=TEAL)
        _caption(slide, Inches(x), Inches(6.02), Inches(5.8), Inches(0.85), body, size=13, align=PP_ALIGN.LEFT)


def _load_metrics() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "logs" / "ppo_walk" / "metrics.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_curve_png(path: Path, rows: list[dict]) -> None:
    """Four learning curves from the saved PPO log."""
    panels = (
        ("reward", "報酬（移動平均）", (31, 78, 95)),
        ("grasp", "把持率（移動平均）", (138, 69, 42)),
        ("ee_dist", "手先と箱の距離（移動平均）", (44, 100, 74)),
        ("policy_loss", "方策 Loss（移動平均）", (70, 78, 92)),
    )
    width, height = 1400, 860
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\YuGothM.ttc", 32)
    small = ImageFont.truetype(r"C:\Windows\Fonts\YuGothM.ttc", 18)
    margin, gap = 36, 28
    panel_w = (width - 2 * margin - gap) // 2
    panel_h = (height - 2 * margin - gap) // 2
    for index, (key, title, color) in enumerate(panels):
        left = margin + (index % 2) * (panel_w + gap)
        top = margin + (index // 2) * (panel_h + gap)
        draw.rounded_rectangle((left, top, left + panel_w, top + panel_h), radius=12, fill=(246, 243, 236))
        draw.text((left + 18, top + 12), title, fill=color, font=font)
        points = [(row["iteration"], row[key]) for row in rows if key in row]
        if len(points) < 2:
            draw.text((left + 18, top + 48), "記録なし", fill=(90, 98, 102), font=small)
            continue
        window = max(21, len(points) // 80)
        if window % 2 == 0:
            window += 1
        half = window // 2
        raw = [item[1] for item in points]
        smoothed = []
        for index in range(len(raw)):
            start = max(0, index - half)
            stop = min(len(raw), index + half + 1)
            smoothed.append(sum(raw[start:stop]) / (stop - start))
        step = max(1, len(points) // 500)
        points = [(points[index][0], smoothed[index]) for index in range(0, len(points), step)]
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_max - y_min < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        plot = (left + 18, top + 42, left + panel_w - 16, top + panel_h - 28)
        draw.text((plot[0], plot[3] + 4), f"{x_min:.0f}", fill=(90, 98, 102), font=small)
        draw.text((plot[2] - 70, plot[3] + 4), f"{x_max:.0f}", fill=(90, 98, 102), font=small)
        draw.text((plot[0], plot[1] - 22), f"{y_max:.3g}", fill=(90, 98, 102), font=small)

        def px(x_value: float, y_value: float) -> tuple[float, float]:
            x_pos = plot[0] + (x_value - x_min) / max(x_max - x_min, 1) * (plot[2] - plot[0])
            y_pos = plot[3] - (y_value - y_min) / (y_max - y_min) * (plot[3] - plot[1])
            return x_pos, y_pos

        line = [px(x_value, y_value) for x_value, y_value in points]
        draw.line(line, fill=color, width=3)
    image.save(path)


def add_curriculum_slide(prs) -> None:
    """Stages after the original pick-and-place, and how weights move forward."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()
    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.0), Inches(0.38))
    write(title, ["段階的な学習"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.48), Inches(12.4), Inches(0.30))
    write(
        subtitle,
        ["箱運びはそのまま残す。前後の段階は、形が同じときだけ重みを引き継ぎ、違うときは覚えた歩行を脚として固定する。"],
        size=14,
        color=MUTED,
        spacing=0,
    )
    y = Inches(0.95)
    h = Inches(6.2)
    card(
        slide,
        Inches(0.28),
        y,
        Inches(6.35),
        h,
        "5つの段階",
        [
            "1  歩行　平地で前後・左右・旋回",
            "　　観測 41、行動は脚 12。最初から学ぶ",
            "2  不整地　同じ観測。歩行の重みから続ける",
            "3  箱運び　いまのピック＆プレース",
            "　　Panda は 68/20、Z1 は 66/19",
            "　　歩行のネットワークとは別の保存",
            "4  歩きながら持つ",
            "　　脚は覚えた歩行。腕と進む速さを学ぶ",
            "　　箱は手に固定。落とし方はまだ学ばない",
            "5  キャッチ　静止 → ゆっくり → ばらす → ノイズ",
            "　　近いレベルから続ける。1 秒保持で成功",
            "次の段階へ進むかは、録画を見て決める",
        ],
        TEAL,
    )
    card(
        slide,
        Inches(6.78),
        y,
        Inches(6.25),
        h,
        "重みの渡し方と、その報酬",
        [
            "形が同じとき、重みをそのまま読む",
            "　　歩行 → 不整地",
            "　　歩きながら持つ → キャッチの各レベル",
            "形が違うとき、脚の方策だけを固定する",
            "　　不整地が無ければ平地の歩行",
            "　　歩行が無ければ決まったトロット",
            "箱運びの 68 次元は、歩行の 41 次元へ入らない",
            "",
            "歩行の点は、指令した速度への一致",
            "　　記録の tracking は 0〜1。1 が一致",
            "キャッチの点は、手をボールへ近づける",
            "　　速度を揃える、指のあいだで保持する",
            "人の点数で、この報酬は置き換えていない",
        ],
        GREEN,
    )


def add_reward_slide(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()
    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.0), Inches(0.38))
    write(title, ["報酬"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.48), Inches(12.4), Inches(0.30))
    write(subtitle, ["1ステップごとに点を足す。置ける・つかむなどの出来事は、その瞬間に一度だけ足す。"], size=14, color=MUTED, spacing=0)

    y = Inches(0.95)
    h = Inches(6.2)
    card(
        slide,
        Inches(0.28),
        y,
        Inches(6.35),
        h,
        "増える点",
        [
            "前進　秒速 0.40 m に近いほど高い。係数 4",
            "　　歩行だけの学習では係数 6",
            "接近　手先が箱に近づく　係数 1.5",
            "　　胴体が箱に近づく　係数 0.4",
            "形　箱の近くで指を閉じる　係数 6",
            "把持　新しくつかんだ瞬間　+5",
            "運搬　つかんだ箱が目標に近い　係数 2",
            "接近距離　1歩で縮めた距離 × 8",
            "　　1歩あたり最大 5 cm まで",
            "設置　目標に置けた瞬間　+10",
            "離す　置く段階で指を開く　係数 4",
            "生存　転んでいなければ毎歩　係数 0.3",
        ],
        GREEN,
    )
    card(
        slide,
        Inches(6.78),
        y,
        Inches(6.25),
        h,
        "減る点と、出すタイミング",
        [
            "転倒　傾き 60° または高さ 0.18 m 未満　−5",
            "横滑り　横速度と旋回が大きいと減点",
            "操作量　関節への指令が大きいと減点　係数 0.02",
            "",
            "段階 0　箱へ歩くとき",
            "　　前進、胴体の接近、生存",
            "段階 1　拾うとき",
            "　　手先の接近、指を閉じる形、把持",
            "段階 2　運ぶとき",
            "　　前進、運搬、生存",
            "段階 3　置くとき",
            "　　設置、指を開く",
            "歩行のみの学習では、把持と設置の点は出さない",
            "歩行の段階は速度の一致。キャッチは接近と保持",
        ],
        CLAY,
    )


def add_loss_slide(prs, curve_png: Path | None) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()
    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.0), Inches(0.38))
    write(title, ["Loss と学習曲線"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.48), Inches(12.4), Inches(0.30))
    write(subtitle, ["更新は、方策 Loss ＋ 0.5 × 価値 Loss − 0.005 × エントロピー。"], size=14, color=MUTED, spacing=0)
    card(
        slide,
        Inches(0.28),
        Inches(0.95),
        Inches(6.15),
        Inches(6.2),
        "3つの Loss",
        [
            "方策 Loss",
            "　良かった行動の確率を上げる",
            "　変化は clip 0.2 までに制限",
            "　ゼロ付近なら更新は小さい",
            "価値 Loss　係数 0.5",
            "　これからの報酬合計の予測誤差",
            "　0.5 ×（実際 − 予測）の二乗",
            "エントロピー　係数 0.005",
            "　行動のばらつきを少し残す",
            "　引き算なので、探索が報酬になる",
            "",
            "曲線の見方",
            "　報酬・把持率は上向きが良い",
            "　手先と箱の距離は下向きが良い",
            "　方策 Loss は 0 の近くに落ち着く",
            "歩行の記録は tracking。1 が指令どおり",
            "右の4枚は箱運びの曲線",
        ],
        TEAL,
    )
    if curve_png is not None and curve_png.is_file():
        slide.shapes.add_picture(str(curve_png), Inches(6.6), Inches(1.15), Inches(6.4), Inches(5.7))
    else:
        note = slide.shapes.add_textbox(Inches(6.7), Inches(2.4), Inches(6.0), Inches(1.2))
        write(note, ["学習記録がまだ無いので、曲線は空です。"], size=16, color=MUTED, spacing=0)


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(10.4), Inches(0.40))
    write(title, ["Pick-and-Carry　学習条件・観測・制御"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.50), Inches(12.6), Inches(0.28))
    write(
        subtitle,
        ["Unitree Go2 ＋ Franka Panda または Unitree Z1　　PPO　　箱運びは Panda 68/20、Z1 66/19（形が違う重みは継続しない）"],
        size=12,
        color=MUTED,
        spacing=0,
    )

    y = Inches(0.88)
    h = Inches(6.38)
    w = Inches(4.18)
    gap = Inches(0.14)
    x0 = Inches(0.28)
    card(
        slide,
        x0,
        y,
        w,
        h,
        "学習条件",
        [
            "ロボット　Go2 に Panda、または Z1（6 軸）",
            "方策　256×256 ELU、出力は −1〜1",
            "更新　16 環境 × 128 step / iteration",
            "最適化　Adam 3e-4、clip 0.2",
            "割引　γ 0.99、λ 0.95、entropy 0.005",
            "時間　dt 0.02 s、1 試行 20 秒",
            "コース　平地 → 不整地 → 拾う → 不整地 → 置く",
            "地面　波 3.5 cm、または段差 4.5 / 6 cm",
            "箱と目標の下は平ら。段は毎回 ±2 cm",
            "箱　一辺 4 cm。拾う 0.55 m、置く 0.45 m",
            "成功　目標の半径 0.28 m、高さ 0.08 m 未満",
            "前進　0.40 m/s を追う。転倒 −5、設置 +10",
            "分け方　歩行のみ／分割／同時 から選択",
            "継続　Panda は logs/ppo_walk、Z1 は ppo_z1",
        ],
        TEAL,
    )
    card(
        slide,
        x0 + w + gap,
        y,
        w,
        h,
        "観測　68",
        [
            "位置は胴体基準。前が +X、上が +Z",
            "脚の角度　　　　12",
            "脚の角速度　　　12　×0.05",
            "重力の向き　　　3",
            "胴体の並進速度　3　×2",
            "胴体の角速度　　3　×0.25",
            "胴体の高さ　　　1　世界座標の z",
            "足の接地　　　　4　接触力 1 N 超",
            "腕の角度　　　　7",
            "腕の角速度　　　7　×0.05",
            "指の開き　　　　1　0〜0.04 m",
            "指の速度　　　　1　×0.05",
            "手先・物資・目標　各 3",
            "把持フラグ 1、段階 one-hot 4",
            "カメラ時は物資の 3 つだけ検知位置",
            "見えない物資は (0, 0, −1)",
        ],
        CLAY,
    )
    card(
        slide,
        x0 + 2 * (w + gap),
        y,
        w,
        h,
        "制御　20",
        [
            "位置制御。PD で目標角度へ追従",
            "脚 12　立ち姿勢 + 出力 × 0.50 rad",
            "　　hip 0、thigh 0.8、calf −1.5",
            "　　kp 80 / kd 4",
            "腕 7　初期姿勢 + 出力 × 1.2 rad",
            "　　(0, 0.9, 0, −1.4, 0, 1.8, 0.7)",
            "　　kp 120 / kd 12",
            "指 1　−1 で閉じ、+1 で開き 0.04 m",
            "　　kp 40 / kd 4",
            "把持　開き 0.012 m 未満、距離 0.12 m",
            "段階　0 歩く　1 拾う　2 運ぶ　3 置く",
            "分割　0・2 は脚、1・3 は腕",
            "歩行のみ　脚だけ。腕は固定、指は開く",
            "同時　脚・腕・指をエピソード中ずっと出力",
        ],
        GREEN,
    )
    add_problem_slide(prs)
    add_curriculum_slide(prs)
    add_reward_slide(prs)
    curve_png = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry_curves.png"
    rows = _load_metrics()
    if rows:
        render_curve_png(curve_png, rows)
    add_loss_slide(prs, curve_png if rows else None)

    out = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry_学習条件.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(out)


if __name__ == "__main__":
    main()
