"""Summary slides for the pick-and-carry training setup (incl. control diagrams)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
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
SOFT_TEAL = RGBColor(0xD7, 0xE4, 0xE8)
SOFT_GREEN = RGBColor(0xD9, 0xE8, 0xDF)
SOFT_CLAY = RGBColor(0xF0, 0xE0, 0xD8)
SOFT_GRAY = RGBColor(0xE8, 0xE4, 0xDC)
ASSETS = Path(__file__).resolve().parent / "assets"
ROBOT_SIM_PNG = ASSETS / "robot_sim_go2_z1.png"


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


def _flow_box(slide, x, y, w, h, title, lines, fill, accent):
    """Rounded content box used in the control flowchart."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.adjustments[0] = 0.08
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = accent
    shape.line.width = Pt(1.25)
    shape.shadow.inherit = False
    head = slide.shapes.add_textbox(x + Inches(0.08), y + Inches(0.06), w - Inches(0.16), Inches(0.28))
    write(head, [title], size=13, bold=True, color=accent, align=PP_ALIGN.LEFT, spacing=0)
    if lines:
        body = slide.shapes.add_textbox(
            x + Inches(0.08),
            y + Inches(0.34),
            w - Inches(0.16),
            h - Inches(0.40),
        )
        write(body, lines, size=11, color=INK, align=PP_ALIGN.LEFT, spacing=1)
    return shape


def _arrow_right(slide, x, y, w=Inches(0.28), h=Inches(0.18), color=MUTED):
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()
    arrow.shadow.inherit = False
    return arrow


def _arrow_down(slide, x, y, w=Inches(0.18), h=Inches(0.24), color=MUTED):
    arrow = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, x, y, w, h)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()
    arrow.shadow.inherit = False
    return arrow


def _clear_slide_shapes(slide) -> None:
    """Remove every shape from a slide (for in-place redraw)."""
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)


def draw_control_flow_slide(slide) -> None:
    """Populate a blank slide with the CPG + analytic-IK control diagram."""
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.08), Inches(12.4), Inches(0.34))
    write(title, ["制御方針（CPG＋IK）"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.40), Inches(12.6), Inches(0.28))
    write(
        subtitle,
        [
            "脚は CPG（4 振動子・対角トロット）が 12 関節へ展開。"
            "腕は解析ヤコビアン IK ベース＋小さな残差。最後は PD。"
        ],
        size=13,
        color=MUTED,
        spacing=0,
    )

    _flow_box(
        slide,
        Inches(3.6),
        Inches(0.72),
        Inches(6.1),
        Inches(0.88),
        "方策（Actor）　行動 22",
        ["CPG パラメータ 15 ＋ 腕残差 Δq 6 ＋ 指残差 Δg 1　（おおよそ −1〜1）"],
        SOFT_TEAL,
        TEAL,
    )
    _arrow_down(slide, Inches(6.45), Inches(1.62), color=TEAL)

    # Two columns that must not overlap: left ≤ 6.4", right ≥ 6.7".
    mid = Inches(6.55)
    _caption(
        slide,
        Inches(0.32),
        Inches(1.88),
        Inches(6.0),
        Inches(0.26),
        "脚レーン（全段階）",
        size=14,
        bold=True,
        color=GREEN,
        align=PP_ALIGN.LEFT,
    )
    _caption(
        slide,
        Inches(6.75),
        Inches(1.88),
        Inches(6.2),
        Inches(0.26),
        "腕レーン（reach / haul / carry / catch）",
        size=14,
        bold=True,
        color=CLAY,
        align=PP_ALIGN.LEFT,
    )

    # Leg: three compact boxes in the left half only.
    leg_y = Inches(2.22)
    lbw, lbh = Inches(1.85), Inches(1.20)
    lgap = Inches(0.22)
    lx = Inches(0.32)
    _flow_box(slide, lx, leg_y, lbw, lbh, "① CPG 15", ["周波数・振幅・持ち上げ", "姿勢・旋回など"], SOFT_GREEN, GREEN)
    _arrow_right(slide, lx + lbw + Inches(0.02), leg_y + Inches(0.50), w=Inches(0.18), color=GREEN)
    lx = lx + lbw + lgap
    _flow_box(slide, lx, leg_y, lbw, lbh, "② 環境内 CPG", ["対角トロットで", "12 関節へ展開"], SOFT_GREEN, GREEN)
    _arrow_right(slide, lx + lbw + Inches(0.02), leg_y + Inches(0.50), w=Inches(0.18), color=GREEN)
    lx = lx + lbw + lgap
    _flow_box(slide, lx, leg_y, lbw, lbh, "③ 脚 PD", ["kp 60 / kd 1.5", "→ Go2 脚 12"], SOFT_GRAY, MUTED)

    # Arm: three boxes in the right half only (same row).
    arm_y = Inches(2.22)
    abw, abh = Inches(1.90), Inches(1.20)
    agap = Inches(0.22)
    ax = Inches(6.75)
    _flow_box(
        slide,
        ax,
        arm_y,
        abw,
        abh,
        "① EE → 解析 IK",
        ["解析ヤコビアン", "で q_ik を解く"],
        SOFT_CLAY,
        CLAY,
    )
    _arrow_right(slide, ax + abw + Inches(0.02), arm_y + Inches(0.50), w=Inches(0.18), color=CLAY)
    ax = ax + abw + agap
    _flow_box(
        slide,
        ax,
        arm_y,
        abw,
        abh,
        "② ＋ 残差",
        ["q_ik + 0.12·Δq", "指 + 0.35·Δg"],
        SOFT_CLAY,
        CLAY,
    )
    _arrow_right(slide, ax + abw + Inches(0.02), arm_y + Inches(0.50), w=Inches(0.18), color=CLAY)
    ax = ax + abw + agap
    _flow_box(
        slide,
        ax,
        arm_y,
        abw,
        abh,
        "③ 腕・指 PD",
        ["腕 80/8　指 20/2", "→ Z1 + グリッパ"],
        SOFT_GRAY,
        MUTED,
    )

    # Walk note: left column only (no overlap with arm).
    note = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.32),
        Inches(3.55),
        Inches(6.05),
        Inches(1.05),
    )
    note.adjustments[0] = 0.08
    note.fill.solid()
    note.fill.fore_color.rgb = WHITE
    note.line.color.rgb = GREEN
    note.line.width = Pt(1.0)
    note.shadow.inherit = False
    write(
        note,
        [
            "歩行（walk）では腕レーンを使わない",
            "マスクは CPG のみ。腕は固定、または学習用にランダム保持。",
        ],
        size=12,
        color=INK,
        spacing=2,
    )
    note.text_frame.margin_left = Inches(0.12)
    note.text_frame.margin_top = Inches(0.14)

    # Arm note: right column.
    arm_note = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(6.75),
        Inches(3.55),
        Inches(6.20),
        Inches(1.05),
    )
    arm_note.adjustments[0] = 0.08
    arm_note.fill.solid()
    arm_note.fill.fore_color.rgb = WHITE
    arm_note.line.color.rgb = CLAY
    arm_note.line.width = Pt(1.0)
    arm_note.shadow.inherit = False
    write(
        arm_note,
        [
            "IK は解析ヤコビアン（数値差分ではない）",
            "EE 目標（到達点・箱・置き場）→ q_ik → 残差で微調整。",
        ],
        size=12,
        color=INK,
        spacing=2,
    )
    arm_note.text_frame.margin_left = Inches(0.12)
    arm_note.text_frame.margin_top = Inches(0.14)

    _ = mid  # documents the left/right split used above
    _box(slide, Inches(0.32), Inches(4.80), Inches(12.65), Inches(2.40), WHITE, line=TEAL)
    _caption(
        slide,
        Inches(0.48),
        Inches(4.90),
        Inches(12.2),
        Inches(0.28),
        "なぜこの分け方か",
        size=15,
        bold=True,
        color=TEAL,
        align=PP_ALIGN.LEFT,
    )
    left = slide.shapes.add_textbox(Inches(0.48), Inches(5.25), Inches(6.0), Inches(1.75))
    write(
        left,
        [
            "脚（CPG）",
            "・12 関節を直接出さず、歩容の「つまみ」15 個にする",
            "・位相は環境が持つので、リズムが崩れにくい",
            "・BC の教師も同じ CPG 空間で教える",
        ],
        size=12,
        color=INK,
        spacing=2,
    )
    right = slide.shapes.add_textbox(Inches(6.80), Inches(5.25), Inches(5.9), Inches(1.75))
    write(
        right,
        [
            "腕（解析 IK＋残差）",
            "・大まかな手先位置は解析 IK が担当（探索を小さく）",
            "・方策は微調整だけ → 学習が安定しやすい",
            "・旧・腕フル指令の重みとは非互換（行動の意味が違う）",
        ],
        size=12,
        color=INK,
        spacing=2,
    )


def add_control_flow_slide(prs) -> None:
    """Diagram: policy → CPG legs / analytic IK+residual arm → PD."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    draw_control_flow_slide(slide)


def add_purpose_slide(prs) -> None:
    """First slide: what we want to achieve (report wording)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.12), Inches(12.4), Inches(0.40))
    write(title, ["最終課題の目的"], size=28, bold=True, spacing=0)

    hero = slide.shapes.add_textbox(Inches(0.40), Inches(0.65), Inches(12.4), Inches(1.15))
    write(
        hero,
        [
            "四足ロボット（Go2）にアーム（Z1）を載せた構成で、",
            "不整地を歩いて「箱」を所定位置まで運ぶ方策を学習すること。",
        ],
        size=20,
        bold=True,
        color=INK,
        spacing=5,
    )

    card(
        slide,
        Inches(0.28),
        Inches(2.05),
        Inches(6.25),
        Inches(5.05),
        "やりたいこと",
        [
            "アーム付き四足で不整地を渡り、箱を拾う",
            "もう一度不整地を渡り、目標へ置く",
            "脚の歩行と腕の操作を一つの方策で扱う",
            "手先制御は、将来のキャッチ課題にもつなげる",
        ],
        TEAL,
    )
    card(
        slide,
        Inches(6.78),
        Inches(2.05),
        Inches(6.25),
        Inches(5.05),
        "難しい点と対応方針",
        [
            "① いきなり箱運びは難しい",
            "　→ 教師 CPG を BC で模倣し、",
            "　　歩行 → 到達 → 箱運びと段階を上げる",
            "",
            "② 脚 12 関節と腕を同時探索すると重い",
            "　→ 脚は CPG、腕は解析 IK＋残差に構造化",
        ],
        CLAY,
    )


def add_robot_model_slide(prs) -> None:
    """Robot model + problem setting on one report slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.4), Inches(0.36))
    write(title, ["ロボットと問題設定"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.46), Inches(12.6), Inches(0.26))
    write(
        subtitle,
        ["Genesis 上の Unitree Go2 ＋ Z1。平地→不整地→拾う→不整地→置く。1 試行 20 秒。"],
        size=13,
        color=MUTED,
        spacing=0,
    )

    if ROBOT_SIM_PNG.is_file():
        slide.shapes.add_picture(str(ROBOT_SIM_PNG), Inches(0.32), Inches(0.90), Inches(7.6), Inches(4.25))
    else:
        note = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(7.0), Inches(1.0))
        write(note, [f"画像がありません: {ROBOT_SIM_PNG.name}"], size=16, color=MUTED, spacing=0)

    card(
        slide,
        Inches(8.15),
        Inches(0.90),
        Inches(4.85),
        Inches(2.85),
        "モデル",
        [
            "胴体　Go2（四足）",
            "脚　3 関節 × 4 ＝ 12",
            "腕　Z1（6 軸＋グリッパ）",
            "手先 TCP を解析 IK の目標に",
        ],
        TEAL,
    )
    card(
        slide,
        Inches(8.15),
        Inches(3.90),
        Inches(4.85),
        Inches(3.20),
        "課題の流れ",
        [
            "出発（平地）→ 不整地",
            "→ 箱を拾う → 不整地",
            "→ 目標へ置く",
            "成功：目標近く・低く置く",
            "失敗：大きく傾くと転倒",
        ],
        CLAY,
    )


def add_cpg_network_slide(prs) -> None:
    """Explain the CPG network: params → 4 oscillators → 12 joints."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.08), Inches(12.4), Inches(0.34))
    write(title, ["CPG ネットワーク（脚のリズム生成）"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.42), Inches(12.6), Inches(0.26))
    write(
        subtitle,
        ["方策は関節を直接出さず、CPG のつまみ 15 個を出す。環境内の 4 振動子が対角トロットで 12 関節へ展開する。"],
        size=12,
        color=MUTED,
        spacing=0,
    )

    _flow_box(
        slide,
        Inches(0.30),
        Inches(0.85),
        Inches(2.55),
        Inches(1.55),
        "方策",
        ["観測 81", "→ CPG パラメータ 15"],
        SOFT_TEAL,
        TEAL,
    )
    _arrow_right(slide, Inches(2.95), Inches(1.45), color=TEAL)
    _flow_box(
        slide,
        Inches(3.35),
        Inches(0.85),
        Inches(3.10),
        Inches(1.55),
        "CPG デコード",
        ["freq / amp / lift…", "dphi・couple・entrain"],
        SOFT_GREEN,
        GREEN,
    )
    _arrow_right(slide, Inches(6.55), Inches(1.45), color=GREEN)
    _flow_box(
        slide,
        Inches(6.95),
        Inches(0.85),
        Inches(3.10),
        Inches(1.55),
        "4 振動子",
        ["FL・FR・RL・RR", "各足が位相 φ を保持"],
        SOFT_CLAY,
        CLAY,
    )
    _arrow_right(slide, Inches(10.15), Inches(1.45), color=CLAY)
    _flow_box(
        slide,
        Inches(10.55),
        Inches(0.85),
        Inches(2.45),
        Inches(1.55),
        "12 関節",
        ["hip/thigh/calf×4", "→ 脚 PD"],
        SOFT_GRAY,
        MUTED,
    )

    # Oscillator network map
    _caption(
        slide,
        Inches(0.30),
        Inches(2.60),
        Inches(7.2),
        Inches(0.28),
        "振動子ネットワーク（対角トロット＋軟結合）",
        size=14,
        bold=True,
        color=GREEN,
        align=PP_ALIGN.LEFT,
    )

    def osc(x, y, name, role, fill, accent):
        sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, Inches(1.70), Inches(1.05))
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
        sh.line.color.rgb = accent
        sh.line.width = Pt(1.75)
        sh.shadow.inherit = False
        _caption(slide, x, y + Inches(0.18), Inches(1.70), Inches(0.30), name, size=14, bold=True, color=accent)
        _caption(slide, x, y + Inches(0.50), Inches(1.70), Inches(0.35), role, size=11, color=INK)

    osc(Inches(0.55), Inches(3.05), "FL", "ペア A　φ", SOFT_GREEN, GREEN)
    osc(Inches(3.00), Inches(3.05), "FR", "ペア B　φ＋π", SOFT_CLAY, CLAY)
    osc(Inches(0.55), Inches(4.45), "RL", "ペア B", SOFT_CLAY, CLAY)
    osc(Inches(3.00), Inches(4.45), "RR", "ペア A", SOFT_GREEN, GREEN)

    # Soft coupling note between diagonals
    couple = slide.shapes.add_textbox(Inches(4.95), Inches(3.15), Inches(2.55), Inches(2.2))
    write(
        couple,
        [
            "軟結合 couple",
            "対角の位相へ引き戻す",
            "",
            "Δab / Δa / Δb",
            "不整地用のずれ",
            "",
            "entrain",
            "接地でスタンスへ引く",
        ],
        size=12,
        color=INK,
        spacing=1,
    )

    card(
        slide,
        Inches(7.70),
        Inches(2.60),
        Inches(5.25),
        Inches(4.25),
        "なぜ CPG か",
        [
            "12 関節を直接学習すると探索が重い",
            "リズム構造を環境側に置き、方策は",
            "　周波数・振幅・旋回などのつまみだけ",
            "教師（CPG）→ BC → PPO と同一の",
            "　行動空間（CPG 15）を共有できる",
            "平坦では硬い対角トロット",
            "不整地では位相適応 5 つで接地をずらす",
            "展開後は sin(φ) で振り、持ち上げは",
            "　relu(−sin(φ))。後退は amp の符号",
        ],
        TEAL,
    )


def add_cpg_ik_detail_slide(prs) -> None:
    """One detailed slide: CPG→which legs, and IK+residual for the arm."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.28), Inches(0.06), Inches(12.6), Inches(0.34))
    write(title, ["CPG と逆運動学（IK）を使った学習　—　詳細"], size=22, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.30), Inches(0.38), Inches(12.6), Inches(0.26))
    write(
        subtitle,
        ["方策は 22 次元。脚は対角トロット CPG（位相オフセット・軟結合・接地引き込み付き）、手先は解析ヤコビアン IK＋残差。"],
        size=12,
        color=MUTED,
        spacing=0,
    )

    # --- Left: bird's-eye leg map ---
    _caption(
        slide,
        Inches(0.28),
        Inches(0.68),
        Inches(6.4),
        Inches(0.28),
        "脚 CPG：どの足にどうつながるか（上から見た対角トロット）",
        size=14,
        bold=True,
        color=GREEN,
        align=PP_ALIGN.LEFT,
    )

    # Body rectangle
    body = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.55), Inches(1.55), Inches(2.6), Inches(2.35))
    body.adjustments[0] = 0.08
    body.fill.solid()
    body.fill.fore_color.rgb = SOFT_GRAY
    body.line.color.rgb = MUTED
    body.line.width = Pt(1.25)
    body.shadow.inherit = False
    _caption(slide, Inches(1.55), Inches(2.35), Inches(2.6), Inches(0.28), "胴体 base", size=12, bold=True, color=MUTED)
    _caption(slide, Inches(1.55), Inches(2.62), Inches(2.6), Inches(0.28), "+X 前＝グリッパ", size=11, color=CLAY)
    _caption(slide, Inches(1.55), Inches(2.88), Inches(2.6), Inches(0.24), "（アームは後ろ寄り）", size=10, color=MUTED)

    # Four legs as boxes: FL FR top, RL RR bottom (viewer facing same as +X up on slide? 
    # Typically top of slide = front. FL left-front, FR right-front.
    def leg_card(x, y, name, phase_txt, fill, accent):
        sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(1.55), Inches(0.95))
        sh.adjustments[0] = 0.1
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
        sh.line.color.rgb = accent
        sh.line.width = Pt(1.5)
        sh.shadow.inherit = False
        _caption(slide, x, y + Inches(0.08), Inches(1.55), Inches(0.28), name, size=13, bold=True, color=accent)
        _caption(slide, x + Inches(0.06), y + Inches(0.38), Inches(1.43), Inches(0.50), phase_txt, size=11, color=INK, align=PP_ALIGN.LEFT)

    # Pair A: FL + RR  → phase φ   (diagonal)
    # Pair B: FR + RL  → phase φ+π
    leg_card(Inches(0.35), Inches(1.10), "FL 左前", "ペア A\nφ", SOFT_GREEN, GREEN)
    leg_card(Inches(3.85), Inches(1.10), "FR 右前", "ペア B\nφ＋π＋Δab", SOFT_CLAY, CLAY)
    leg_card(Inches(0.35), Inches(3.10), "RL 左後", "ペア B\nφ＋π＋Δab＋Δb", SOFT_CLAY, CLAY)
    leg_card(Inches(3.85), Inches(3.10), "RR 右後", "ペア A\nφ＋Δa", SOFT_GREEN, GREEN)

    # Diagonal connectors (thin lines via rectangles as dashes)
    diag1 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.85), Inches(2.05), Inches(2.0), Inches(0.04))
    diag1.rotation = 38.0
    diag1.fill.solid()
    diag1.fill.fore_color.rgb = GREEN
    diag1.line.fill.background()
    diag2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.85), Inches(2.95), Inches(2.0), Inches(0.04))
    diag2.rotation = -38.0
    diag2.fill.solid()
    diag2.fill.fore_color.rgb = CLAY
    diag2.line.fill.background()

    note_l = slide.shapes.add_textbox(Inches(0.30), Inches(4.20), Inches(5.9), Inches(1.15))
    write(
        note_l,
        [
            "各足 → hip / thigh / calf の 3 関節（合計 12）。脚ごとに位相を保持",
            "軟結合で対角構造へ引き戻し、Δab/Δa/Δb で不整地のずれを学習可能",
            "接地立ち上がりで位相をスタンス（0）へ引き込み（entrain）",
            "sin(φ) で振り、持ち上げは relu(−sin(φ))。後退は amp 符号で、すね位相を π",
        ],
        size=11,
        color=INK,
        spacing=1,
    )

    # --- Right: CPG params + IK ---
    _caption(
        slide,
        Inches(6.50),
        Inches(0.68),
        Inches(6.5),
        Inches(0.28),
        "方策が出す CPG 15 パラメータ（リズム 10 ＋ 位相適応 5）",
        size=14,
        bold=True,
        color=TEAL,
        align=PP_ALIGN.LEFT,
    )
    params = slide.shapes.add_textbox(Inches(6.50), Inches(0.98), Inches(6.5), Inches(2.55))
    write(
        params,
        [
            "0–9  freq / amp / thigh / calf / lift / hip_lat / hip_yaw / turn / bias",
            "10 dphi_ab　ペア B の π からのずれ（対角のタイミング）",
            "11 dphi_a　　ペア A 内（RR 相対 FL）",
            "12 dphi_b　　ペア B 内（RL 相対 FR）",
            "13 couple　　対角への軟結合の強さ（弱いほど独立）",
            "14 entrain　接地時にスタンス位相へ引き込む強さ",
            "教師は Δ=0・couple≈1・entrain≈0（硬い対角トロット）",
            "→ LegCPG が 12 関節指令へ展開 → 脚 PD（kp60/kd1.5）",
        ],
        size=11,
        color=INK,
        spacing=1,
    )

    _caption(
        slide,
        Inches(6.50),
        Inches(3.55),
        Inches(6.5),
        Inches(0.28),
        "腕の逆運動学（IK）＋残差　reach / haul / carry / catch",
        size=14,
        bold=True,
        color=CLAY,
        align=PP_ALIGN.LEFT,
    )
    ik_box = slide.shapes.add_textbox(Inches(6.50), Inches(3.85), Inches(6.5), Inches(2.05))
    write(
        ik_box,
        [
            "① EE 目標を選ぶ（到達点・箱・目標・ボール予測など）",
            "② 解析ヤコビアン IK（減衰最小二乗）でベース姿勢 q_ik を解く",
            "③ 方策の腕 Δq 6 を ×0.12 rad して加算　q = q_ik + 0.12·Δq",
            "④ 指は段階の開閉ベース + 残差 ×0.35　→ 腕・指 PD",
            "歩行（walk）では腕レーンは使わない（CPG のみ／腕は固定や乱数保持）",
            "BC・PPO は同じ 22 次元（CPG15＋腕6＋指1）を共有",
        ],
        size=11,
        color=INK,
        spacing=1,
    )

    # Bottom bar
    _box(slide, Inches(0.28), Inches(6.55), Inches(12.75), Inches(0.75), SOFT_TEAL, line=TEAL)
    foot = slide.shapes.add_textbox(Inches(0.40), Inches(6.62), Inches(12.5), Inches(0.60))
    write(
        foot,
        [
            "まとめ　方策は歩容つまみ＋位相適応（15）と手先微調整を学習。"
            "平坦では硬い対角トロット、不整地では Δ・couple・entrain で接地タイミングをずらす。"
            "腕の大きな移動は IK、細かい合わせ込みは残差 RL。",
        ],
        size=12,
        color=INK,
        spacing=1,
    )


def add_control_compare_slide(prs) -> None:
    """Side-by-side: previous joint residual vs current structured control."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.0), Inches(0.38))
    write(title, ["制御の比較（以前 → 今回）"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.48), Inches(12.4), Inches(0.28))
    write(
        subtitle,
        ["同じ観測 81。行動だけ 19（脚 12 直接）から 17（CPG 15 + 腕残差）へ。構造を環境側に寄せた。"],
        size=13,
        color=MUTED,
        spacing=0,
    )

    card(
        slide,
        Inches(0.28),
        Inches(0.95),
        Inches(6.25),
        Inches(5.95),
        "以前　行動 19",
        [
            "脚　12　立ち姿勢からの差分を直接指令",
            "腕　6　初期姿勢からの差分を直接指令",
            "指　1　開閉を直接指令",
            "",
            "特徴",
            "・歩容の周期性をゼロから探す",
            "・手先の大きな移動も方策の探索任せ",
            "・次元が多く、模倣教師も 19 次元",
            "",
            "向いていたこと",
            "・単純な追従や短い課題",
            "",
            "つらかったこと",
            "・長い歩行でばらつきが関節端へ張り付く",
            "・reach 以降の腕の探索が広い",
        ],
        MUTED,
    )
    card(
        slide,
        Inches(6.78),
        Inches(0.95),
        Inches(6.25),
        Inches(5.95),
        "今回　行動 22",
        [
            "脚　CPG 15　→ 環境が 12 関節へ展開",
            "腕　IK ベース + 残差 ×0.12 rad",
            "指　段階ベース開閉 + 残差 ×0.35 rad",
            "",
            "特徴",
            "・歩容の形は CPG、方策はパラメータ調整",
            "・手先の大まかな位置は IK、方策は微調整",
            "・BC / PPO が同じ 22 次元を共有",
            "",
            "向いていること",
            "・歩行暖機（教師 CPG → 模倣 → PPO）",
            "・到達・箱運び・キャッチの手先合わせ",
            "",
            "注意",
            "・旧 19 次元や旧・腕フル指令の重みは使えない",
            "・walk では腕残差は方策外（マスク）",
        ],
        GREEN,
    )


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
    path = Path(__file__).resolve().parents[1] / "logs" / "ppo_z1" / "metrics.jsonl"
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
        ["歩行を土台に、到達とキャッチを足す。箱運びはそのまま残す。全段階が観測 81・行動 22（CPG+腕）なので、重みを引き継ぐ。"],
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
        "段階",
        [
            "1  歩行　立つ・歩く・旋回。平地から始め、地形へ広げる",
            "2  到達　歩行の重みから。空中の点へ手先を届かせる",
            "3  歩きながら持つ　到達の重みから",
            "4  箱運び　持ち歩きの重みから",
            "5  キャッチ　到達の重みから",
            "　　静止 → ゆっくり → ばらす → 飛ぶ → ノイズ",
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
            "同じ観測と行動で、直前の重みを初期値にする",
            "　　歩行 → 到達 → 持ち歩き → 箱運び",
            "　　到達 → キャッチ（静止から順に）",
            "歩行は腕を勝手に動かして学ぶ（腕の揺れに強くする）",
            "腕は Unitree Z1 だけ",
            "今までの Panda の保存は使えない",
            "",
            "歩行の点は、指令した速度への一致（legged_gym 標準）",
            "　　足を上げる時間、胴の揺れ、力の大きさは罰",
            "　　記録の tracking は 0〜1。1 が一致",
            "到達・キャッチの点は、手をボールへ近づける",
            "　　速度を揃える、指のあいだで保持する",
            "人の点数で、この報酬は置き換えていない",
        ],
        GREEN,
    )


def add_technique_slide(prs) -> None:
    """How training is staged, rewarded, and allowed to step back."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()
    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.0), Inches(0.38))
    write(title, ["学習の工夫"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.48), Inches(12.4), Inches(0.30))
    write(
        subtitle,
        ["一度に全部を覚えさせない。できた能力は次へ渡し、成績が落ちたら易しい課題へ戻してから上げる。"],
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
        "覚え方",
        [
            "歩行と把持を同時にゼロから学ばせない",
            "段階は、歩行 → 到達 → 持ち歩き → 箱運び",
            "　　到達 → キャッチ（5 段階のボール）",
            "同じ観測 81・行動 22（CPG + 腕 IK 残差）です",
            "　　直前の段階の重みを初期値にする",
            "脚は CPG、腕は IK ベース＋小さな残差",
            "腕は Unitree Z1 だけ",
            "旧 19 次元や腕フル指令の保存は使えない",
        ],
        TEAL,
    )
    card(
        slide,
        Inches(6.78),
        y,
        Inches(6.25),
        h,
        "報酬と、難しさの戻し方",
        [
            "成功の瞬間だけでは学ばない",
            "毎ステップ、接近・速度の一致・生存で点を出す",
            "箱運びは段階で出す点を変える",
            "　　歩くときは前進、拾うときは把持、置くときは設置",
            "歩行の記録は tracking。1 が指令どおり",
            "",
            "8 回続けて成績が低いと、一段易しくする",
            "　　歩行の地形 → 半分 → 平地（平地から始める）",
            "　　ノイズ → 飛ぶ → ばらす → ゆっくり → 静止",
            "8 回続けて回復すると、選んだ課題まで上げる",
            "メニューでは、上の段階の重みで",
            "　　易しい課題を学び直せる",
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
            "歩行は速度の一致。到達・キャッチは接近と保持",
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
    write(subtitle, ["更新は、方策 Loss ＋ 1.0 × 価値 Loss − 0.01 × エントロピー。"], size=14, color=MUTED, spacing=0)
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
            "価値 Loss　係数 1.0",
            "　これからの報酬合計の予測誤差",
            "　0.5 ×（実際 − 予測）の二乗",
            "エントロピー　係数 0.01",
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


def add_obs_action_slide(prs) -> None:
    """Observation vector and action / control quantities."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.4), Inches(0.36))
    write(title, ["観測・制御量"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.46), Inches(12.6), Inches(0.28))
    write(
        subtitle,
        ["全段階で観測 81・行動 22 を共有。位置は胴体基準（前が +X）。脚指令は CPG 展開後の 12 関節も観測に入れる。"],
        size=13,
        color=MUTED,
        spacing=0,
    )

    y = Inches(0.88)
    h = Inches(6.35)
    w = Inches(6.25)
    gap = Inches(0.20)
    card(
        slide,
        Inches(0.28),
        y,
        w,
        h,
        "観測　81",
        [
            "脚の角度・角速度　　各 12",
            "重力方向 3、胴体並進・角速度 各 3、高さ 1",
            "足の接地　　4（接触力 1 N 超）",
            "腕の角度・角速度　各 6、指の開閉・速度 各 1",
            "手先・物資・目標　各 3（胴体基準）",
            "把持フラグ 1、段階 one-hot 4",
            "速度指令 3（キャッチ時は予測位置）",
            "直前の脚指令 12（CPG が展開した関節）",
        ],
        CLAY,
    )
    card(
        slide,
        Inches(0.28) + w + gap,
        y,
        w,
        h,
        "制御量　行動 22 → 関節目標",
        [
            "方策出力",
            "　CPG 15　リズムのつまみ",
            "　腕 Δq 6　IK への残差（×0.12 rad）",
            "　指 Δg 1　開閉残差（×0.35）",
            "",
            "環境側で組み立て",
            "　CPG → 脚 12 関節　PD kp60/kd1.5",
            "　IK + Δq → 腕 6　PD kp80/kd8",
            "　段階ベース + Δg → 指　PD kp20/kd2",
            "",
            "学習設定（抜粋）",
            "　16 環境 × 128 step、γ0.99、clip0.2",
            "　1 試行 20 秒、dt 0.02 s",
        ],
        GREEN,
    )


def add_learning_flow_slide(prs) -> None:
    """BC first, then curriculum RL stages."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.4), Inches(0.36))
    write(title, ["学習の流れ（BC → カリキュラム）"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.46), Inches(12.6), Inches(0.28))
    write(
        subtitle,
        ["まず教師 CPG を BC でまね、そのあと PPO で段階を上げる。各段階の重みを次の初期値にする。"],
        size=13,
        color=MUTED,
        spacing=0,
    )

    # Horizontal flow
    boxes = [
        (Inches(0.30), "① 教師 CPG", "前進・左右旋回", SOFT_TEAL, TEAL),
        (Inches(2.85), "② BC", "同じ 3 技能を模倣", SOFT_GREEN, GREEN),
        (Inches(5.40), "③ 歩行 PPO", "指令追従を強化", SOFT_CLAY, CLAY),
        (Inches(7.95), "④ 到達", "手先を空中点へ", SOFT_TEAL, TEAL),
        (Inches(10.50), "⑤ 箱運び", "拾う・運ぶ・置く", SOFT_GREEN, GREEN),
    ]
    for x, head, body, fill, accent in boxes:
        _flow_box(slide, x, Inches(0.95), Inches(2.35), Inches(1.35), head, [body], fill, accent)
    for x in (Inches(2.55), Inches(5.10), Inches(7.65), Inches(10.20)):
        _arrow_right(slide, x, Inches(1.45), w=Inches(0.26), color=MUTED)

    card(
        slide,
        Inches(0.28),
        Inches(2.60),
        Inches(6.25),
        Inches(4.55),
        "模倣（教師 → BC）",
        [
            "教師は開ループ CPG（対角トロット）",
            "演示は前進・左旋回・右旋回のみ",
            "BC が同じ行動空間（CPG 15）を学習",
            "ここまでで「歩ける初期方策」を得る",
        ],
        TEAL,
    )
    card(
        slide,
        Inches(6.78),
        Inches(2.60),
        Inches(6.25),
        Inches(4.55),
        "カリキュラム（PPO）",
        [
            "歩行　速度指令への一致を強化学習",
            "到達　歩行重みから。手先を点へ",
            "箱運び　拾って目標へ置く",
            "（任意）キャッチは到達重みから",
            "観測 81・行動 22 が全段階で共通",
            "次段階へ進むかは録画を見て判断",
        ],
        GREEN,
    )


def add_conclusion_slide(prs) -> None:
    """Closing reflection: struggles, ambitions, compute limits."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.12), Inches(12.4), Inches(0.40))
    write(title, ["結論・今後に向けて"], size=28, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.55), Inches(12.6), Inches(0.28))
    write(
        subtitle,
        ["今回できたことと、足りなかった点を整理する。"],
        size=14,
        color=MUTED,
        spacing=0,
    )

    card(
        slide,
        Inches(0.28),
        Inches(1.05),
        Inches(4.15),
        Inches(5.95),
        "学習で苦労したこと",
        [
            "歩行の安定化に時間がかかった",
            "腕を同時に動かすと探索が重い",
            "段階のつなぎ（重み引き継ぎ）で",
            "　性能が落ちやすい",
            "報酬設計と難度の戻しが繊細",
            "録画を見ながらの判断が必要",
        ],
        TEAL,
    )
    card(
        slide,
        Inches(4.58),
        Inches(1.05),
        Inches(4.15),
        Inches(5.95),
        "やりたかった高度なタスク",
        [
            "飛来ボールのキャッチ完成",
            "持ち歩き〜設置までの一気通貫",
            "より速い指令・荒い不整地",
            "実機寄りのセンサノイズ耐性",
            "複数箱・長いコースの連続搬送",
        ],
        CLAY,
    )
    card(
        slide,
        Inches(8.88),
        Inches(1.05),
        Inches(4.15),
        Inches(5.95),
        "計算リソースの制約",
        [
            "手元 GPU では並列環境数・",
            "　反復数が限られる",
            "reach 以降は IK で step が重い",
            "長時間カリキュラムを回し切れず",
            "高度な段階まで十分に学習できず",
            "→ 解析 IK などで高速化しつつ",
            "　段階を絞って提出に至った",
        ],
        GREEN,
    )


def add_reward_method_slide(prs) -> None:
    """Compact reward + training tricks for the report."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = PAPER
    background.line.fill.background()
    title = slide.shapes.add_textbox(Inches(0.32), Inches(0.10), Inches(12.4), Inches(0.36))
    write(title, ["報酬と学習の工夫"], size=24, bold=True, spacing=0)
    subtitle = slide.shapes.add_textbox(Inches(0.34), Inches(0.46), Inches(12.6), Inches(0.26))
    write(
        subtitle,
        ["成功の瞬間だけでなく毎ステップで導く。難しすぎたら一段易しく戻す。"],
        size=13,
        color=MUTED,
        spacing=0,
    )
    card(
        slide,
        Inches(0.28),
        Inches(0.90),
        Inches(6.25),
        Inches(6.20),
        "主な報酬",
        [
            "前進　0.40 m/s への一致",
            "接近　手先→箱、箱→目標",
            "把持・設置　成功時に加点",
            "転倒　大きく減点",
            "歩行段階は速度トラッキング中心",
            "到達は手先の接近・保持が中心",
        ],
        TEAL,
    )
    card(
        slide,
        Inches(6.78),
        Inches(0.90),
        Inches(6.25),
        Inches(6.20),
        "工夫",
        [
            "段階分け（歩行→到達→箱運び）",
            "直前段階の重みを初期値にする",
            "観測 81・行動 22 を全段階で共有",
            "成績が続くと難度を戻す／上げる",
            "脚 CPG・腕 解析 IK＋残差で探索を縮小",
        ],
        GREEN,
    )


def build_report_presentation() -> Presentation:
    """Lean deck for the written report (no duplicate / outdated slides)."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    add_purpose_slide(prs)
    add_robot_model_slide(prs)
    add_control_flow_slide(prs)
    add_cpg_network_slide(prs)
    add_obs_action_slide(prs)
    add_learning_flow_slide(prs)
    add_reward_method_slide(prs)
    curve_png = Path(__file__).resolve().parents[1] / "output" / "PickAndCarry_curves.png"
    rows = _load_metrics()
    if rows:
        render_curve_png(curve_png, rows)
    add_loss_slide(prs, curve_png if rows else None)
    add_conclusion_slide(prs)
    return prs


def main() -> None:
    # Full reference deck (kept for development / long appendix).
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    add_purpose_slide(prs)
    add_robot_model_slide(prs)
    add_control_flow_slide(prs)
    add_obs_action_slide(prs)
    add_learning_flow_slide(prs)
    add_cpg_network_slide(prs)
    add_cpg_ik_detail_slide(prs)
    add_control_compare_slide(prs)
    add_problem_slide(prs)
    add_curriculum_slide(prs)
    add_technique_slide(prs)
    add_reward_slide(prs)
    curve_png = Path(__file__).resolve().parents[1] / "output" / "PickAndCarry_curves.png"
    rows = _load_metrics()
    if rows:
        render_curve_png(curve_png, rows)
    add_loss_slide(prs, curve_png if rows else None)

    out_desktop = Path(__file__).resolve().parents[1] / "output" / "PickAndCarry_学習条件.pptx"
    out_repo = Path(__file__).resolve().parent / "PickAndCarry_学習条件.pptx"
    out_desktop.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_desktop)
    prs.save(out_repo)
    print(out_desktop)
    print(out_repo)

    # Report-oriented 改版 (deduplicated, clearer order).
    report = build_report_presentation()
    out_report = Path(__file__).resolve().parents[1] / "output" / "PickAndCarry_学習条件_改版.pptx"
    report.save(out_report)
    print(out_report)


if __name__ == "__main__":
    main()
