"""Record the stand and sit poses against the heights those tasks ask for.

The viewer stays closed. One mp4 shows four beats: the crouch as placed, the
same crouch after the feet meet the ground, the legs extended as far as the
action range allows, then the standing placement and the idle crouch used by sit.

    .venv\\Scripts\\python scripts\\show_pose_heights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pick_and_carry.config import EnvConfig
from pick_and_carry.env import PickAndCarryEnv, _rgb_uint8

FPS = 25
DT = 0.02


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _output_path() -> Path:
    desktop = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry_pose.mp4"
    if desktop.parent.is_dir():
        return desktop
    path = ROOT / "logs" / "PickAndCarry_pose.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pack(env: PickAndCarryEnv, leg: torch.Tensor) -> torch.Tensor:
    arm = env.default_arm.expand(1, -1)
    fingers = torch.full((1, len(env.finger_names)), 0.5 * env.cfg.gripper_open, device=env.device)
    pose = torch.cat((leg.view(1, -1), arm, fingers), dim=-1)
    return pose.clamp(env.dof_lower, env.dof_upper) if hasattr(env, "dof_lower") else pose


def _leg_target(env: PickAndCarryEnv, kind: str) -> torch.Tensor:
    if kind == "crouch":
        leg = env.crouch_leg.clone()
    elif kind == "stand":
        leg = env.default_leg.clone()
    elif kind == "extend":
        leg = env.crouch_leg.clone()
        leg[1::3] = leg[1::3] - env.cfg.leg_action_scale
        leg[2::3] = leg[2::3] + env.cfg.leg_action_scale
    else:
        raise ValueError(kind)
    return leg.clamp(env.leg_lower, env.leg_upper)


def _hold(env: PickAndCarryEnv, kind: str) -> None:
    target = _pack(env, _leg_target(env, kind))
    env.robot.control_dofs_position(target, env.dof_idx)


def _place(env: PickAndCarryEnv, kind: str, height: float) -> None:
    leg = _leg_target(env, kind)
    target = _pack(env, leg)
    base = env.init_base_pos.view(1, 3).clone()
    base[:, 2] = height
    quat = env.init_base_quat.view(1, 4).expand(1, 4)
    env.robot.set_pos(base, zero_velocity=True)
    env.robot.set_quat(quat, zero_velocity=True)
    env.robot.set_dofs_position(target, env.dof_idx, zero_velocity=True)
    env.robot.control_dofs_position(target, env.dof_idx)
    env._read_state()


def _frame(env: PickAndCarryEnv) -> np.ndarray:
    rgb, *_ = env.camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)
    return _rgb_uint8(rgb)


def _compose(image: np.ndarray, title: str, detail: str, height: float) -> np.ndarray:
    picture = Image.fromarray(image)
    width, pic_h = picture.size
    bar_h = 118
    canvas = Image.new("RGB", (width, pic_h + bar_h), (18, 20, 24))
    canvas.paste(picture, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    body_font = _font(22)
    draw.text((24, pic_h + 14), title, font=title_font, fill=(236, 236, 236))
    draw.text((24, pic_h + 56), detail, font=body_font, fill=(186, 198, 210))
    draw.text((24, pic_h + 86), f"いまの胴体の高さ  {height:.2f} m", font=body_font, fill=(255, 214, 120))

    gauge_x = width - 168
    gauge_top = pic_h + 18
    gauge_bot = pic_h + bar_h - 16
    draw.rectangle((gauge_x, gauge_top, gauge_x + 10, gauge_bot), fill=(70, 74, 82))

    def mark(level: float, label: str, color: tuple[int, int, int]) -> None:
        span = gauge_bot - gauge_top
        y = gauge_bot - int(round(np.clip(level, 0.0, 0.50) / 0.50 * span))
        draw.line((gauge_x - 6, y, gauge_x + 16, y), fill=color, width=2)
        draw.text((gauge_x + 22, y - 12), label, font=_font(16), fill=color)

    mark(0.40, "0.40 立つ", (120, 190, 255))
    mark(0.28, "0.28 座る", (140, 220, 160))
    span = gauge_bot - gauge_top
    y = gauge_bot - int(round(np.clip(height, 0.0, 0.50) / 0.50 * span))
    draw.line((gauge_x - 8, y, gauge_x + 18, y), fill=(255, 214, 120), width=3)
    return np.ascontiguousarray(np.asarray(canvas))


def _write(writer, env: PickAndCarryEnv, title: str, detail: str) -> float:
    image = _compose(_frame(env), title, detail, float(env.base_pos[0, 2]))
    import av

    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    frame.pts = writer["index"]
    writer["index"] += 1
    for packet in writer["stream"].encode(frame):
        writer["container"].mux(packet)
    return float(env.base_pos[0, 2])


def _still(writer, env: PickAndCarryEnv, kind: str, height: float, title: str, detail: str, seconds: float) -> None:
    frames = max(1, int(round(seconds * FPS)))
    for _ in range(frames):
        _place(env, kind, height)
        _write(writer, env, title, detail)


def _simulate(writer, env: PickAndCarryEnv, kind: str, title: str, detail: str, seconds: float) -> float:
    _hold(env, kind)
    steps = max(1, int(round(seconds / DT)))
    stride = max(1, int(round(1.0 / (FPS * DT))))
    last = float(env.base_pos[0, 2])
    for step in range(steps):
        env.scene.step()
        env._read_state()
        last = float(env.base_pos[0, 2])
        if step % stride == 0:
            _write(writer, env, title, detail)
    return last


def main() -> None:
    import av

    out = _output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    print("シーンを組んでいます。3D の窓は開きません。", flush=True)
    env = PickAndCarryEnv(
        cfg=EnvConfig(
            num_envs=1,
            backend="gpu",
            show_viewer=False,
            enable_camera=True,
            terrain=False,
            curriculum="stand",
            control_split="both",
            seed=0,
            arm="z1",
        )
    )
    env.camera.set_pose(pos=(1.5, -1.7, 0.62), lookat=(0.0, 0.0, 0.22))
    far = torch.tensor([[40.0, 40.0, 0.5]], dtype=torch.float32, device=env.device)
    env.cube.set_pos(far, zero_velocity=True)
    env.goal_marker.set_pos(far, zero_velocity=True)

    container = av.open(str(out), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = 960
    stream.height = 540 + 118
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "20"}
    writer = {"container": container, "stream": stream, "index": 0}
    try:
        _still(
            writer,
            env,
            "crouch",
            0.30,
            "立つ　置いた瞬間のしゃがみ",
            "プログラムは胴体を 0.30 m に置く。足はまだ地面についていない。",
            1.5,
        )
        settled = _simulate(
            writer,
            env,
            "crouch",
            "立つ　同じしゃがみのまま足が着く",
            "操作 0 はこのしゃがみ。足が着くと胴体は 0.23 m。",
            2.5,
        )
        print(f"crouch settled {settled:.3f}", flush=True)
        extended = _simulate(
            writer,
            env,
            "extend",
            "立つ　操作を一杯まで伸ばす",
            "届いたのは 0.32 m。立つ成功は 0.34〜0.46 m。",
            3.0,
        )
        print(f"extend settled {extended:.3f}", flush=True)
        _still(
            writer,
            env,
            "stand",
            0.42,
            "座る　置いた瞬間の立位",
            "プログラムは胴体を 0.42 m に置く。足はまだ地面についていない。",
            1.5,
        )
        stood = _simulate(
            writer,
            env,
            "stand",
            "座る　同じ立位のまま足が着く",
            "この関節が支えられる高さは 0.31 m。",
            2.5,
        )
        print(f"stand settled {stood:.3f}", flush=True)
        sat = _simulate(
            writer,
            env,
            "crouch",
            "座る　操作 0 のしゃがみ",
            "足が着くと 0.23 m。座る成功は 0.24〜0.32 m。",
            3.0,
        )
        print(f"sit idle settled {sat:.3f}", flush=True)
    finally:
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
        env.close()
    print(f"video {out}", flush=True)
    try:
        import os

        os.startfile(out)
    except Exception:
        pass


if __name__ == "__main__":
    main()
