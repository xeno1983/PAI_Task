"""Run records and learning-curve pages.

Training appends one JSON line per iteration. The curve page is a single HTML
file, so it opens in a browser without a metrics server.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

CHARTS = (
    ("reward", "報酬", "1 iteration の平均報酬"),
    ("grasp", "把持率", "箱をつかんでいた割合"),
    ("ee_dist", "手先と箱の距離", "メートル。小さいほど手が届いている"),
    ("success", "設置率", "その時点で終了した試行のうち、置けた割合"),
)


def write_run(path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"recorded_at": datetime.now(timezone.utc).isoformat(), **info}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_metrics(path: Path) -> list[dict]:
    """Read metrics. A repeated iteration keeps the later line."""
    if not path.is_file():
        return []
    by_step: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        by_step[int(row["iteration"])] = row
    return [by_step[key] for key in sorted(by_step)]


def _series(rows: list[dict], key: str) -> list[tuple[float, float]]:
    points = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        points.append((float(row["iteration"]), float(value)))
    return points


def _smooth(points: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    if window <= 1 or len(points) < 2:
        return []
    smoothed = []
    values = [point[1] for point in points]
    for index, (step, _) in enumerate(points):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        smoothed.append((step, sum(chunk) / len(chunk)))
    return smoothed


def _polyline(points: list[tuple[float, float]], bounds: tuple[float, float, float, float], size: tuple[int, int]) -> str:
    if not points:
        return ""
    xmin, xmax, ymin, ymax = bounds
    width, height = size
    pad_x, pad_y = 46, 28
    span_x = max(xmax - xmin, 1e-9)
    span_y = max(ymax - ymin, 1e-9)

    def sx(x: float) -> float:
        return pad_x + (x - xmin) / span_x * (width - pad_x - 12)

    def sy(y: float) -> float:
        return height - pad_y - (y - ymin) / span_y * (height - pad_y - 16)

    commands = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    return commands


def _chart(title: str, caption: str, points: list[tuple[float, float]]) -> str:
    width, height = 760, 230
    if not points:
        return (
            f"<section><h2>{title}</h2><p class='empty'>{caption}</p>"
            "<p class='empty'>この指標はまだ記録されていません。</p></section>"
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.08 or 0.05
    bounds = (min(xs), max(xs), ymin - pad, ymax + pad)
    raw = _polyline(points, bounds, (width, height))
    window = 20 if len(points) >= 20 else max(2, len(points) // 5)
    trend = _polyline(_smooth(points, window), bounds, (width, height))
    last = points[-1][1]
    return f"""<section>
      <h2>{title}</h2>
      <p>{caption}。最新 {last:.4g}</p>
      <svg viewBox="0 0 {width} {height}" role="img">
        <polyline points="{raw}" class="raw"/>
        <polyline points="{trend}" class="trend"/>
        <text x="8" y="20">{ymax:.3g}</text>
        <text x="8" y="{height - 8}">{ymin:.3g}</text>
        <text x="{width - 70}" y="{height - 8}">{int(xs[-1])}</text>
      </svg>
    </section>"""


def render_curves(rows: list[dict], run: dict | None = None) -> str:
    """Return a self-contained HTML page for the recorded iterations."""
    run = run or {}
    bits = []
    if run:
        bits.append(
            "<ul class='meta'>"
            + "".join(f"<li><b>{key}</b> {value}</li>" for key, value in run.items() if key != "recorded_at")
            + "</ul>"
        )
    if not rows:
        bits.append("<p class='empty'>学習記録がまだありません。メニューの 2 で学習すると、ここに曲線が出ます。</p>")
    else:
        bits.append(f"<p>{len(rows)} iteration を記録しています。細い線が各回、太い線が移動平均です。</p>")
        for key, title, caption in CHARTS:
            bits.append(_chart(title, caption, _series(rows, key)))
    body = "\n".join(bits)
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<title>Pick-and-Carry 学習曲線</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; margin: 24px; color: #1c1c1c; background: #f6f4ef; }}
  h1 {{ font-size: 22px; margin-bottom: 8px; }}
  section {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin: 12px 0; }}
  h2 {{ font-size: 16px; margin: 0 0 4px; }}
  p {{ margin: 0 0 8px; color: #444; }}
  .empty {{ color: #666; }}
  .meta {{ display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 0; list-style: none; }}
  .meta li {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 4px 8px; }}
  svg {{ width: 100%; height: auto; }}
  .raw {{ fill: none; stroke: #c5c1b8; stroke-width: 1.5; }}
  .trend {{ fill: none; stroke: #0b6e4f; stroke-width: 2.5; }}
  text {{ font-size: 12px; fill: #666; }}
</style>
</head>
<body>
<h1>Pick-and-Carry 学習曲線</h1>
{body}
</body>
</html>
"""


def load_tensorboard_metrics(log_dir: Path) -> list[dict]:
    """Build iteration rows from TensorBoard scalars when metrics.jsonl is absent."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []
    if not log_dir.is_dir():
        return []
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = {
        "rollout/mean_reward": "reward",
        "rollout/grasp_rate": "grasp",
        "rollout/ee_dist": "ee_dist",
        "rollout/success_rate": "success",
        "rollout/fps": "fps",
    }
    by_step: dict[int, dict] = {}
    for tag, key in tags.items():
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            row = by_step.setdefault(int(event.step), {"iteration": int(event.step)})
            row[key] = float(event.value)
    return [by_step[key] for key in sorted(by_step)]
