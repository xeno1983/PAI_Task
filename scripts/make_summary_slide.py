"""One-slide summary of the pick-and-carry training setup."""

from __future__ import annotations

from pathlib import Path

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
        ["Unitree Go2 ＋ 背中の Franka Panda　　PPO　　観測 68 次元・制御 20 次元（この寸法のまま重みを継続）"],
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
            "ロボット　Go2 の胴体に Panda を固定",
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
            "継続　logs/ppo_walk の保存重みから再開",
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

    out = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry_学習条件.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(out)


if __name__ == "__main__":
    main()
