"""Click a top-down course to place a supply and its drop point, then run the policy.

The Genesis viewer is not opened. After both clicks, inference records an mp4.

    .venv\\Scripts\\python scripts\\infer_gui.py --log-dir logs/ppo_z1 --record out.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pick_and_carry.terrain import COURSE_X, COURSE_Y, GOAL_FLAT, PICKUP_FLAT, ROUGH_1, ROUGH_2

PAD = 28
CANVAS_W = 820
CANVAS_H = 420


def canvas_point(x: float, y: float) -> tuple[float, float]:
    u = (x - COURSE_X[0]) / (COURSE_X[1] - COURSE_X[0])
    v = (y - COURSE_Y[0]) / (COURSE_Y[1] - COURSE_Y[0])
    return PAD + u * (CANVAS_W - 2 * PAD), CANVAS_H - PAD - v * (CANVAS_H - 2 * PAD)


def world_point(px: float, py: float) -> tuple[float, float]:
    u = (px - PAD) / (CANVAS_W - 2 * PAD)
    v = (CANVAS_H - PAD - py) / (CANVAS_H - 2 * PAD)
    x = COURSE_X[0] + min(1.0, max(0.0, u)) * (COURSE_X[1] - COURSE_X[0])
    y = COURSE_Y[0] + min(1.0, max(0.0, v)) * (COURSE_Y[1] - COURSE_Y[0])
    return x, y


def latest_iteration(log_dir: Path) -> int:
    best = 0
    if not log_dir.is_dir():
        return 0
    for path in log_dir.glob("ppo_*.pt"):
        stem = path.stem
        if stem.startswith("ppo_") and stem[4:].isdigit():
            best = max(best, int(stem[4:]))
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Click supply and drop points, then run inference.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/ppo_z1"))
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--terrain-amplitude", type=float, default=None)
    parser.add_argument("--terrain-steps", action="store_true")
    parser.add_argument("--arm", choices=("z1",), default="z1")
    parser.add_argument("--control-split", choices=("split", "walk", "both"), default="split")
    parser.add_argument("--wrist-camera", action="store_true")
    parser.add_argument("--ros", action="store_true", help="While playing, accept rviz2 clicks as the drop point.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit("tkinter が無いため、クリック画面を開けません。") from exc

    iteration = latest_iteration(ROOT / args.log_dir if not args.log_dir.is_absolute() else args.log_dir)
    log_dir = args.log_dir if args.log_dir.is_absolute() else ROOT / args.log_dir

    root = tk.Tk()
    root.title(f"推論  iteration {iteration}")
    root.geometry(f"{CANVAS_W + 20}x{CANVAS_H + 150}")
    status = tk.StringVar(value="先に物資の位置をクリックしてください。次のクリックが運び先です。")
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x", padx=12, pady=(10, 0))
    tk.Label(
        root,
        text=f"使う重み: iteration {iteration}    橙が物資、緑が運び先。学習で使った平地を薄く示しています。",
        anchor="w",
    ).pack(fill="x", padx=12)

    canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="#efe6d6", highlightthickness=0)
    canvas.pack(padx=10, pady=8)

    def band(x0: float, x1: float, color: str) -> None:
        ax, _ = canvas_point(x0, COURSE_Y[1])
        bx, by = canvas_point(x1, COURSE_Y[0])
        _, top = canvas_point(x0, COURSE_Y[1])
        canvas.create_rectangle(ax, top, bx, by, fill=color, outline="")

    band(COURSE_X[0], COURSE_X[1], "#f4efe4")
    band(*ROUGH_1, "#c4b49a")
    band(*ROUGH_2, "#c4b49a")
    band(*PICKUP_FLAT, "#e7f0dd")
    band(*GOAL_FLAT, "#e7f0dd")
    start, _ = canvas_point(0.0, 0.0)
    _, sy = canvas_point(0.0, 0.0)
    canvas.create_oval(start - 6, sy - 6, start + 6, sy + 6, fill="#444444", outline="")
    canvas.create_text(start + 16, sy - 12, text="出発", anchor="w", fill="#333333")

    marks: dict[str, tuple[float, float]] = {}
    drawn: dict[str, int] = {}

    def redraw(kind: str, x: float, y: float, color: str) -> None:
        if kind in drawn:
            canvas.delete(drawn[kind])
        px, py = canvas_point(x, y)
        drawn[kind] = canvas.create_oval(px - 8, py - 8, px + 8, py + 8, fill=color, outline="#222222", width=2)

    def on_click(event) -> None:
        x, y = world_point(event.x, event.y)
        if "object" not in marks:
            marks["object"] = (x, y)
            redraw("object", x, y, "#e07030")
            status.set(f"物資 ({x:.2f}, {y:.2f})。次に運び先をクリックしてください。")
            return
        if "goal" not in marks:
            marks["goal"] = (x, y)
            redraw("goal", x, y, "#2e9a55")
            ox, oy = marks["object"]
            status.set(f"物資 ({ox:.2f}, {oy:.2f})  →  運び先 ({x:.2f}, {y:.2f})")
            start_button.configure(state="normal")
            return
        ox, oy = marks["object"]
        gx, gy = marks["goal"]
        object_far = (ox - x) ** 2 + (oy - y) ** 2
        goal_far = (gx - x) ** 2 + (gy - y) ** 2
        kind = "object" if object_far >= goal_far else "goal"
        marks[kind] = (x, y)
        redraw(kind, x, y, "#e07030" if kind == "object" else "#2e9a55")
        ox, oy = marks["object"]
        gx, gy = marks["goal"]
        status.set(f"物資 ({ox:.2f}, {oy:.2f})  →  運び先 ({gx:.2f}, {gy:.2f})")

    canvas.bind("<Button-1>", on_click)

    def run_inference() -> None:
        ox, oy = marks["object"]
        gx, gy = marks["goal"]
        command = [
            sys.executable,
            "-m",
            "pick_and_carry.train_ppo",
            "--play",
            "--resume",
            "latest",
            "--num-envs",
            "1",
            "--seconds",
            str(args.seconds),
            "--log-dir",
            str(log_dir),
            "--record",
            str(args.record),
            "--object-xy",
            f"{ox:.4f},{oy:.4f}",
            "--goal-xy",
            f"{gx:.4f},{gy:.4f}",
            "--control-split",
            args.control_split,
            "--arm",
            args.arm,
        ]
        if args.terrain_steps:
            command.append("--terrain-steps")
        if args.terrain_amplitude is not None:
            command.extend(["--terrain-amplitude", str(args.terrain_amplitude)])
        if args.wrist_camera:
            command.append("--wrist-camera")
        if args.ros:
            command.extend(["--ros"])
        if args.ros:
            status.set("推論を実行しています。シミュレーション画面が開き、rviz2 の Publish Point で運び先を動かせます。")
        else:
            status.set("推論を実行しています。シーンの作成に数分かかります。")
        start_button.configure(state="disabled")
        root.update()
        result = subprocess.run(command, cwd=ROOT)
        start_button.configure(state="normal")
        if result.returncode == 0:
            status.set(f"動画を書きました: {args.record}")
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/c", "start", "", str(args.record)], cwd=ROOT)
        else:
            status.set(f"推論が失敗しました。終了コード {result.returncode}")

    start_button = tk.Button(root, text="この配置で推論する", state="disabled", command=run_inference)
    start_button.pack(pady=(0, 12))
    root.mainloop()


if __name__ == "__main__":
    main()
