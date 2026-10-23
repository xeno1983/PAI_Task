"""Build a short PPT explaining Git / MLflow / DVC / MinIO with real UI screenshots."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

FONT = "Yu Gothic"
INK = RGBColor(0x1C, 0x28, 0x2C)
MUTED = RGBColor(0x4A, 0x58, 0x5C)
PAPER = RGBColor(0xF6, 0xF3, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEAL = RGBColor(0x1F, 0x4E, 0x5F)
GREEN = RGBColor(0x2C, 0x64, 0x4A)
CLAY = RGBColor(0x8A, 0x45, 0x2A)
SOFT = RGBColor(0xE8, 0xE4, 0xDC)

ASSETS = Path(__file__).resolve().parent / "assets" / "dvc_mlflow_ppt"
LAUNCHER = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"


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


def write(shape, lines, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, spacing=6) -> None:
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    for index, line in enumerate(lines):
        paragraph = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        paragraph.alignment = align
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(spacing)
        run = paragraph.add_run()
        run.text = line
        paint(run, size, bold=bold, color=color)


def blank(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = PAPER
    bg.line.fill.background()
    return slide


def title_bar(slide, title: str, subtitle: str | None = None) -> None:
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.18), Inches(12.4), Inches(0.7))
    write(box, [title], size=26, bold=True, color=WHITE, spacing=0)
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.45), Inches(1.05), Inches(12.4), Inches(0.4))
        write(sub, [subtitle], size=13, color=MUTED, spacing=0)


def add_picture_fit(slide, path: Path, left, top, width, height) -> None:
    if not path.is_file():
        missing = slide.shapes.add_textbox(left, top, width, height)
        write(missing, [f"(missing screenshot: {path.name})"], size=12, color=CLAY)
        return
    frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    frame.adjustments[0] = 0.04
    frame.fill.solid()
    frame.fill.fore_color.rgb = WHITE
    frame.line.color.rgb = SOFT
    pad = Inches(0.1)
    box_w = float(width - 2 * pad)
    box_h = float(height - 2 * pad)
    with Image.open(path) as img:
        iw, ih = img.size
    scale = min(box_w / iw, box_h / ih)
    pic_w = iw * scale
    pic_h = ih * scale
    pic_left = left + pad + (box_w - pic_w) / 2
    pic_top = top + pad + (box_h - pic_h) / 2
    slide.shapes.add_picture(str(path), pic_left, pic_top, width=pic_w, height=pic_h)


def flow_box(slide, left, top, width, height, head, lines, accent=TEAL, fill=WHITE, head_size=15, size=12):
    """Rounded card with colored header strip."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.adjustments[0] = 0.07
    card.fill.solid()
    card.fill.fore_color.rgb = fill
    card.line.color.rgb = accent
    card.line.width = Pt(1.25)
    strip_h = Inches(0.46)
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, strip_h)
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent
    strip.line.fill.background()
    write(strip, [head], size=head_size, bold=True, color=WHITE, align=PP_ALIGN.CENTER, spacing=0)
    strip.text_frame.margin_top = Inches(0.04)
    body = slide.shapes.add_textbox(left + Inches(0.1), top + strip_h + Inches(0.04), width - Inches(0.2), height - strip_h - Inches(0.08))
    write(body, lines, size=size, color=INK, spacing=3)


def arrow_right(slide, left, top, width=Inches(0.4), height=Inches(0.4), color=MUTED):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()


def arrow_down(slide, left, top, width=Inches(0.4), height=Inches(0.4), color=MUTED):
    a = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()


def tag(slide, left, top, width, text, color=CLAY, size=12):
    t = slide.shapes.add_textbox(left, top, width, Inches(0.32))
    write(t, [text], size=size, bold=True, color=color, spacing=0)


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1 title
    s = blank(prs)
    hero = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    hero.fill.solid()
    hero.fill.fore_color.rgb = TEAL
    hero.line.fill.background()
    box = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.5), Inches(2.5))
    write(
        box,
        [
            "Git × MLflow × DVC × MinIO",
            "Pick-and-Carry での実験記録と重み共有",
        ],
        size=32,
        bold=True,
        color=WHITE,
        spacing=12,
    )
    foot = s.shapes.add_textbox(Inches(0.8), Inches(5.4), Inches(11.5), Inches(1.0))
    write(
        foot,
        [
            "実機の画面（MinIO Console / MLflow UI / DVC status）を使った使い方まとめ",
            "ログイン: MinIO Console http://127.0.0.1:9001  /  MLflow http://127.0.0.1:5001",
        ],
        size=14,
        color=RGBColor(0xD7, 0xE4, 0xE8),
        spacing=4,
    )

    # 2 architecture
    s = blank(prs)
    title_bar(s, "全体像：役割分担", "学習を速くする道具ではなく、あとから困らないための棚分け")
    roles = [
        (Inches(0.45), "Git", GREEN, ["コード・設定", ".dvc ポインタ", "設計図"]),
        (Inches(3.55), "MLflow", TEAL, ["パラメータ", "メトリクス曲線", "実験ノート"]),
        (Inches(6.65), "DVC", CLAY, ["大きい重みの版", "best.pt の選定", "荷物の伝票"]),
        (Inches(9.75), "MinIO", MUTED, ["S3 互換倉庫", "pai-dvc / pai-mlflow", "実ファイル"]),
    ]
    for x, name, accent, lines in roles:
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.7), Inches(2.9), Inches(3.6))
        card.adjustments[0] = 0.08
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.fill.background()
        head = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.7), Inches(2.9), Inches(0.7))
        head.fill.solid()
        head.fill.fore_color.rgb = accent
        head.line.fill.background()
        write(head, [name], size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER, spacing=0)
        body = s.shapes.add_textbox(x + Inches(0.2), Inches(2.6), Inches(2.5), Inches(2.4))
        write(body, lines, size=16, color=INK, spacing=10)
    note = s.shapes.add_textbox(Inches(0.45), Inches(5.6), Inches(12.4), Inches(1.2))
    write(
        note,
        [
            "学習中の生データは logs/（作業机）。区切りだけ DVC publish → MinIO push。",
            "MinIO は DB ではない。オブジェクトストレージ（ファイル倉庫）です。",
        ],
        size=14,
        color=MUTED,
        spacing=4,
    )

    # 2a user journey (what the user does, step by step)
    s = blank(prs)
    title_bar(s, "ユーザの使い方：全体フロー", "起動スクリプト → 学習 → 区切りで共有 → 復元・推論 → 実機へ配信（すべてメニューから）")
    steps = [
        ("① 学習する", TEAL, ["起動スクリプト", "PickAndCarry_run", "→ 学習メニュー", "", "自動で記録:", "logs/ に .pt", "MLflow に曲線"]),
        ("② 区切りで共有", CLAY, ["メニュー 9", "1 MinIO 起動", "3 最良を publish", "  → push", "", "(4 は最新)"]),
        ("③ 保管される", GREEN, ["Git: .dvc", "（ポインタだけ）", "", "MinIO: pai-dvc", "（重みの実体）", "", "Console で確認"]),
        ("④ 復元・推論", TEAL, ["メニュー 9", "5 pull → restore", "6 最良で推論", "  （best.pt を play）", "", "別 PC / 再開に"]),
        ("⑤ 実機へ配信", MUTED, ["メニュー 9 → 9", "Mender Artifact", "（.mender）を作成", "", "Hosted Mender に", "アップロード → Go2"]),
    ]
    box_w, box_h, gap, x0, y0 = Inches(2.15), Inches(3.5), Inches(0.42), Inches(0.45), Inches(1.75)
    for i, (head, accent, lines) in enumerate(steps):
        x = x0 + i * (box_w + gap)
        flow_box(s, x, y0, box_w, box_h, head, lines, accent=accent, size=13)
        if i < len(steps) - 1:
            arrow_right(s, x + box_w + Inches(0.02), y0 + box_h / 2 - Inches(0.2), width=gap - Inches(0.04))
    side = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(5.5), Inches(12.4), Inches(1.45))
    side.adjustments[0] = 0.1
    side.fill.solid()
    side.fill.fore_color.rgb = SOFT
    side.line.fill.background()
    write(
        side,
        [
            "いつでも（学習中でも後からでも）: MLflow UI（http://127.0.0.1:5001）で曲線・パラメータを見て、どの ラン / 段階を最良として共有するか判断する。",
            "① の学習は DVC / MinIO を使わない（速度に影響しない）。共有したいときだけ ② 以降を実行する。",
        ],
        size=14,
        color=INK,
        spacing=6,
    )
    side.text_frame.margin_left = Inches(0.25)
    side.text_frame.margin_top = Inches(0.15)

    # 2b data flow (where files / records go)
    s = blank(prs)
    title_bar(s, "処理フロー図：データはどこへ流れるか", "3 本の流れ：実験の記録（MLflow）／重みの共有（DVC+MinIO）／実機への配信（Mender）")
    rows = [
        (
            "A. 実験の記録（自動）",
            TEAL,
            [
                ("学習", ["train_ppo", "metrics.jsonl / ppo_*.pt", "を logs/ に出力"]),
                ("MLflow に記録", ["params / metrics を", "logs/mlflow/mlflow.db", "(sqlite) へ"]),
                ("MLflow UI", ["http://127.0.0.1:5001", "曲線・Runs 比較"]),
                ("人が判断", ["どの段階が良いか", "→ 次の B で共有"]),
            ],
        ),
        (
            "B. 重みの共有（区切りで手動）",
            CLAY,
            [
                ("logs/ の重み", ["ppo_*.pt", "metrics で最良を選定"]),
                ("dvc_sync publish", ["メニュー 9 → 3 / 4", "best.pt を DVC 管理に"]),
                ("保管", ["MinIO pai-dvc: 実体", "Git: .dvc ポインタ"]),
                ("取り出し", ["9 → 5 pull → restore", "9 → 6 best.pt で推論"]),
            ],
        ),
        (
            "C. 実機への配信（必要なとき）",
            MUTED,
            [
                ("best.pt", ["DVC から resolve", "（B の成果物）"]),
                ("Artifact 作成", ["メニュー 9 → 9", "mender_pack_model.py", "→ .mender"]),
                ("Hosted Mender", ["Artifact を", "アップロード → Deploy"]),
                ("Go2 実機", ["/data/pai/models/", "active/best.pt に配置"]),
            ],
        ),
    ]
    bw, bh, bgap, bx0 = Inches(2.6), Inches(1.4), Inches(0.5), Inches(0.7)
    row_y = [Inches(1.55), Inches(3.35), Inches(5.15)]
    for (label, accent, boxes), y in zip(rows, row_y):
        tag(s, bx0, y - Inches(0.02), Inches(8), label, color=accent, size=13)
        for j, (head, lines) in enumerate(boxes):
            x = bx0 + j * (bw + bgap)
            flow_box(s, x, y + Inches(0.32), bw, bh - Inches(0.1), head, lines, accent=accent, head_size=14, size=12)
            if j < len(boxes) - 1:
                arrow_right(s, x + bw + Inches(0.05), y + Inches(0.32) + (bh - Inches(0.1)) / 2 - Inches(0.18), width=bgap - Inches(0.1), height=Inches(0.36))
    # cross-row hint
    hint = s.shapes.add_textbox(Inches(0.7), Inches(6.95), Inches(12), Inches(0.4))
    write(hint, ["A の記録は自動。B・C は「共有したい」「実機に載せたい」ときだけ、起動スクリプトのメニュー 9 から実行。"], size=12, color=MUTED, spacing=0)

    # 2c decision guide
    s = blank(prs)
    title_bar(s, "やりたいこと → どこを触る？", "迷ったらこの表。操作はすべて 起動スクリプト → メニュー 9 か MLflow UI")
    guide = [
        ("学習の曲線を比較したい", "MLflow UI（:5001）", TEAL),
        ("最良の重みを仲間と共有したい", "9 → 1（MinIO 起動）→ 9 → 3（publish → push）", CLAY),
        ("共有された重みで動かしたい", "9 → 6（pull → resolve → play）", GREEN),
        ("別 PC で学習を再開したい", "git pull → 9 → 5（pull → restore）", GREEN),
        ("保管状況を目で確認したい", "9 → 8（MinIO Console）→ pai-dvc", MUTED),
        ("実機 Go2 に最良モデルを配りたい", "9 → 9（Mender Artifact）→ Hosted Mender", MUTED),
    ]
    for k, (want, act, accent) in enumerate(guide):
        y = Inches(1.65) + k * Inches(0.86)
        q = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), y, Inches(5.0), Inches(0.68))
        q.adjustments[0] = 0.2
        q.fill.solid()
        q.fill.fore_color.rgb = WHITE
        q.line.color.rgb = accent
        write(q, [want], size=16, bold=True, color=INK, align=PP_ALIGN.CENTER, spacing=0)
        arrow_right(s, Inches(5.7), y + Inches(0.14), width=Inches(0.55), height=Inches(0.4), color=accent)
        a = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.4), y, Inches(6.3), Inches(0.68))
        a.adjustments[0] = 0.2
        a.fill.solid()
        a.fill.fore_color.rgb = accent
        a.line.fill.background()
        write(a, [act], size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER, spacing=0)
        for shp in (q, a):
            shp.text_frame.margin_top = Inches(0.1)

    # 3 flow
    s = blank(prs)
    title_bar(s, "いつ何を使うか", "ホットパスでは使わず、区切り・引っ越し・提出で使う")
    steps = [
        "1. 学習 → logs/<stage>/ppo_*.pt + metrics.jsonl + MLflow 記録",
        "2. 区切り → メニュー 9 → 3（最良）または 4（最新）で publish → push",
        "3. 確認 → MinIO Console の pai-dvc（実体） / Git の .dvc（ポインタ）",
        "4. 復元・推論 → pull → restore、または 9 → 6 で best.pt を play",
        "5. 曲線確認 → MLflow UI（実験ノート）。重み本体は DVC/MinIO",
    ]
    box = s.shapes.add_textbox(Inches(0.7), Inches(1.8), Inches(12), Inches(4.5))
    write(box, steps, size=20, color=INK, spacing=16)

    # 4 MinIO login
    s = blank(prs)
    title_bar(s, "MinIO Console：ログイン", "http://127.0.0.1:9001   user / pass = minioadmin（.env と同じ）")
    add_picture_fit(s, ASSETS / "minio_login.png", Inches(1.2), Inches(1.6), Inches(10.8), Inches(5.4))

    # 5 MinIO buckets
    s = blank(prs)
    title_bar(s, "MinIO：バケット一覧（実画面）", "pai-dvc = DVC の重み倉庫　／　pai-mlflow = MLflow artifact 用")
    add_picture_fit(s, ASSETS / "minio_pai_dvc.png", Inches(0.5), Inches(1.55), Inches(12.3), Inches(5.5))

    # 6 MinIO hashed files
    s = blank(prs)
    title_bar(s, "MinIO の中身：DVC の content-addressed 保管", "pai-dvc / files / md5/ … ハッシュ名で実体を置く（Git には載せない）")
    add_picture_fit(s, ASSETS / "minio_pai_dvc_files.png", Inches(0.5), Inches(1.55), Inches(12.3), Inches(5.5))

    # 7 DVC
    s = blank(prs)
    title_bar(s, "DVC：ポインタと最良選定", "Git には .dvc、実体は MinIO。publish --select best で metrics から選ぶ")
    add_picture_fit(s, ASSETS / "dvc_status.png", Inches(0.5), Inches(1.55), Inches(7.4), Inches(5.2))
    side = s.shapes.add_textbox(Inches(8.2), Inches(1.8), Inches(4.6), Inches(4.8))
    write(
        side,
        [
            "メニュー 9",
            "3  最良を publish → push",
            "4  最新を publish → push",
            "6  最良で推論（best.pt）",
            "",
            "CLI 例",
            "dvc_sync.py publish \\",
            "  --log-dir logs\\... \\",
            "  --select best --push",
            "",
            "resolve で best.pt のパス表示",
        ],
        size=15,
        color=INK,
        spacing=6,
    )

    # 8 MLflow home
    s = blank(prs)
    title_bar(s, "MLflow：実験ノート（実画面）", "起動: mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db")
    add_picture_fit(s, ASSETS / "mlflow_home.png", Inches(0.5), Inches(1.55), Inches(12.3), Inches(5.5))

    # 9 MLflow runs UI
    s = blank(prs)
    title_bar(s, "MLflow：pick-and-carry の Runs（UI）", "曲線・パラメータ比較はここ。大きい .pt の共有は DVC/MinIO 側")
    add_picture_fit(s, ASSETS / "mlflow_experiment.png", Inches(0.5), Inches(1.55), Inches(12.3), Inches(5.5))

    # 10 MLflow runs UI (another angle / columns)
    s = blank(prs)
    title_bar(
        s,
        "MLflow：学習済み Runs 一覧（UI スナップショット）",
        "pick-and-carry 実験の walk / reach / carry / catch など実ラン",
    )
    add_picture_fit(s, ASSETS / "mlflow_runs_table.png", Inches(0.4), Inches(1.5), Inches(12.5), Inches(5.5))

    # 11 MLflow run detail UI
    s = blank(prs)
    title_bar(s, "MLflow：ラン詳細（UI スナップショット）", "params / metrics。方策 .pt 本体の共有は DVC / MinIO")
    add_picture_fit(s, ASSETS / "mlflow_run_detail.png", Inches(0.4), Inches(1.5), Inches(12.5), Inches(5.5))

    # 12 MLflow charts UI
    s = blank(prs)
    title_bar(s, "MLflow：メトリクス表示（UI スナップショット）", "ラン詳細画面のメトリクス／チャート領域")
    add_picture_fit(s, ASSETS / "mlflow_metric_curves.png", Inches(0.4), Inches(1.5), Inches(12.5), Inches(5.5))

    # 13 checklist
    s = blank(prs)
    title_bar(s, "最短チェックリスト", "起動スクリプト → メニュー 9")
    lines = [
        "□ 9 → 1  MinIO 起動（Docker Desktop も必要なら起動）",
        "□ 9 → 2  初回のみ setup（pip + DVC remote）",
        "□ 9 → 3  学習済み段階の最良重みを publish → push",
        "□ Console で pai-dvc に objects が増えたことを確認",
        "□ 9 → 6  または resolve + --play で最良を推論",
        "□ MLflow UI で曲線を見る（メトリクスは別系統）",
        "",
        "詳細手順: docs/DVC_MINIO.md",
    ]
    box = s.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.5), Inches(5.0))
    write(box, lines, size=20, color=INK, spacing=12)

    return prs


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    prs = build()
    out_repo = Path(__file__).resolve().parent / "DVC_MLflow_MinIO使い方.pptx"
    out_desk = LAUNCHER / "DVC_MLflow_MinIO使い方.pptx"
    LAUNCHER.mkdir(parents=True, exist_ok=True)
    for target in (out_repo, out_desk):
        try:
            prs.save(target)
            print(f"saved {target}")
        except OSError:
            # File is open in PowerPoint (locked): write a sibling copy instead.
            alt = target.with_name(target.stem + "_ui" + target.suffix)
            try:
                prs.save(alt)
            except OSError:
                alt = target.with_name(target.stem + "_flow" + target.suffix)
                prs.save(alt)
            print(f"locked: {target.name}; saved {alt}")


if __name__ == "__main__":
    main()
