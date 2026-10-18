"""Replay saved PPO snapshots on the same scene and write one comparison video.

Run from the repository root:

    .venv\\Scripts\\python scripts\\compare_snapshots.py --output %USERPROFILE%\\Desktop\\PickAndCarry_compare.mp4

Panels are ordered from the earliest snapshot to the latest, left to right and
then top to bottom. Every panel starts from the same object and goal.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import av
import numpy as np
import torch

from pick_and_carry.agents import ActorCritic
from pick_and_carry.arms import get_arm
from pick_and_carry.compare import draw_label, even_size, select_evenly, snapshot_label, tile_frames
from pick_and_carry.config import EnvConfig
from pick_and_carry.env import PickAndCarryEnv

PHASE_LABELS = ("不整地→物体", "拾う", "不整地→目標", "放す")
# Side view from the start pad to the goal pad.
COMPARE_CAMERA_POS = (2.0, -4.6, 2.6)
COMPARE_CAMERA_LOOKAT = (2.0, 0.0, 0.15)
COMPARE_CAMERA_FOV = 58
COMPARE_CAMERA_RES = (640, 360)


def parse_args():
    parser = argparse.ArgumentParser(description="Make a side-by-side video of PPO snapshots.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/ppo"))
    parser.add_argument("--output", type=Path, default=None, help="Comparison mp4. Defaults to the Desktop.")
    parser.add_argument("--seconds", type=float, default=16.0, help="Simulation seconds shown for every snapshot.")
    parser.add_argument("--panels", type=int, default=4, help="How many snapshots to include, spread from first to last.")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0, help="Same scene for every panel.")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--terrain-amplitude", type=float, default=None)
    parser.add_argument("--terrain-steps", action="store_true")
    parser.add_argument("--arm", choices=("z1",), default="z1")
    parser.add_argument("--control-split", choices=("split", "walk", "both"), default="split")
    parser.add_argument("--wrist-camera", action="store_true")
    return parser.parse_args()


def default_output() -> Path:
    folder = Path.home() / "Desktop" / "起動スクリプト" / "PickAndCarry"
    if folder.is_dir():
        return folder / "PickAndCarry_compare.mp4"
    desktop = Path.home() / "Desktop"
    if desktop.is_dir():
        return desktop / "PickAndCarry_compare.mp4"
    return Path("logs/ppo/compare.mp4")


def load_snapshots(log_dir: Path, expected_obs: int, expected_act: int, arm: str) -> tuple[list[dict], int]:
    records: list[dict] = []
    skipped = 0
    for path in sorted(log_dir.glob("ppo_*.pt")):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        if int(blob.get("obs_dim", -1)) != expected_obs or int(blob.get("act_dim", -1)) != expected_act or str(blob.get("arm", "")) != arm or "model" not in blob:
            skipped += 1
            continue
        records.append(
            {
                "path": path,
                "iteration": int(blob.get("iteration", 0)),
                "model": blob["model"],
            }
        )
    records.sort(key=lambda item: (item["iteration"], item["path"].name))
    return records, skipped


def as_rgb_uint8(rgb) -> np.ndarray:
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    image = np.asarray(rgb)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise RuntimeError(f"camera frame has shape {image.shape}")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        scale = 255.0 if float(np.max(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class _Mp4:
    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.path = path
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "veryfast", "crf": "23"}
        self._index = 0

    def write(self, image: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        frame.pts = self._index
        self._index += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()


def read_frames(path: Path):
    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            yield np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()


def record_panel(env: PickAndCarryEnv, agent: ActorCritic, iteration: int, seconds: float, fps: int, path: Path) -> dict:
    torch.manual_seed(env.cfg.seed)
    obs = env.reset()
    steps = max(1, int(round(seconds / env.cfg.dt)))
    stride = max(1, int(round(1.0 / (fps * env.cfg.dt))))
    label = snapshot_label(iteration)
    writer: _Mp4 | None = None
    grasped = False
    placed = False
    phase = int(env.phase[0])
    try:
        for step in range(steps):
            if step % stride == 0:
                rgb, _, _, _ = env.camera.render(rgb=True, force_render=True)
                frame = draw_label(as_rgb_uint8(rgb), label)
                if writer is None:
                    writer = _Mp4(path, frame.shape[1], frame.shape[0], fps)
                writer.write(frame)
            with torch.no_grad():
                action = agent.act_mean(obs)
            obs, _, _, info = env.step(action)
            grasped = grasped or bool(env.grasped[0])
            placed = placed or bool(info["success"][0])
            phase = int(env.phase[0])
    finally:
        if writer is not None:
            writer.close()
    return {
        "iteration": iteration,
        "base_x": float(env.base_pos[0, 0]),
        "phase": PHASE_LABELS[phase] if 0 <= phase < len(PHASE_LABELS) else str(phase),
        "grasped": grasped,
        "placed": placed,
    }


def stitch(panel_paths: list[Path], output: Path, fps: int) -> int:
    readers = [read_frames(path) for path in panel_paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: _Mp4 | None = None
    count = 0
    try:
        while True:
            panels = []
            for reader in readers:
                try:
                    panels.append(next(reader))
                except StopIteration:
                    panels = []
                    break
            if not panels:
                break
            tiled = even_size(tile_frames(panels))
            if writer is None:
                writer = _Mp4(output, tiled.shape[1], tiled.shape[0], fps)
            writer.write(tiled)
            count += 1
    finally:
        if writer is not None:
            writer.close()
        for reader in readers:
            reader.close()
    return count


def main() -> None:
    args = parse_args()
    if args.panels < 1:
        raise SystemExit("--panels は 1 以上にしてください。")
    if args.fps < 1:
        raise SystemExit("--fps は 1 以上にしてください。")
    output = args.output or default_output()
    from pick_and_carry.video_paths import make_video_run_dir, video_task_name

    arm = get_arm(args.arm)
    records, skipped = load_snapshots(args.log_dir, arm.num_obs, arm.num_actions, arm.name)
    # Nest under log_dir/videos/{task}/compare_iter…/ even when --output is a flat desktop path.
    requested = Path(output)
    if "videos" not in requested.parts:
        last_iter = int(records[-1]["iteration"]) if records else 0
        run_dir = make_video_run_dir(
            args.log_dir,
            task=video_task_name("carry"),
            kind="compare",
            iteration=last_iter,
            extra={"panels": args.panels},
        )
        output = run_dir / requested.name
        print(f"compare → {output}", flush=True)
    if skipped:
        print(f"サイズが違う重みを {skipped} 件スキップしました。", flush=True)
    if not records:
        print(
            f"{args.log_dir.resolve()} に、このタスクの重みがありません。"
            "先に学習すると、ここから比較動画を作れます。",
            flush=True,
        )
        raise SystemExit(1)
    chosen = select_evenly(records, args.panels)
    names = ", ".join(snapshot_label(item["iteration"]) for item in chosen)
    print(f"比較する重み: {names}", flush=True)
    print("左上から、学習の早い順に並びます。物体と目標の位置は全パネルで同じです。", flush=True)

    cfg = EnvConfig(
        num_envs=1,
        backend=args.backend,
        show_viewer=False,
        enable_camera=True,
        camera_res=COMPARE_CAMERA_RES,
        camera_pos=COMPARE_CAMERA_POS,
        camera_lookat=COMPARE_CAMERA_LOOKAT,
        camera_fov=COMPARE_CAMERA_FOV,
        seed=args.seed,
        terrain_steps=args.terrain_steps,
        terrain_amplitude=0.035 if args.terrain_amplitude is None else args.terrain_amplitude,
        control_split=args.control_split,
        arm=args.arm,
        wrist_camera=args.wrist_camera,
    )
    env = PickAndCarryEnv(cfg=cfg)
    agent = ActorCritic(env.num_obs, env.num_actions).to(env.device)
    panel_paths: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix="pac_compare_") as temp_dir:
            temp = Path(temp_dir)
            for index, item in enumerate(chosen):
                agent.load_state_dict(item["model"])
                panel_path = temp / f"panel_{index:02d}.mp4"
                print(f"録画中 {snapshot_label(item['iteration'])} ({item['path'].name})", flush=True)
                stats = record_panel(env, agent, item["iteration"], args.seconds, args.fps, panel_path)
                grasp = "した" if stats["grasped"] else "していない"
                placed = "置いた" if stats["placed"] else "置いていない"
                print(
                    f"  終了時 x={stats['base_x']:.2f}  段階={stats['phase']}  "
                    f"把持={grasp}  設置={placed}",
                    flush=True,
                )
                panel_paths.append(panel_path)
            frames = stitch(panel_paths, output, args.fps)
    finally:
        env.close()
    print(f"video: {output.resolve()} ({frames} frames)", flush=True)


if __name__ == "__main__":
    main()
