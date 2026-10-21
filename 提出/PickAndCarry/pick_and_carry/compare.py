"""Choose training snapshots and lay their frames out for a comparison video.

The simulator is not imported here, so the selection and the frame layout can
be tested without a GPU.
"""

from __future__ import annotations

import numpy as np

# 5x7 glyphs. Each row is five bits, left to right.
_GLYPHS = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "e": ("00000", "00000", "01110", "10101", "11111", "10000", "01111"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "t": ("00100", "00100", "01110", "00100", "00100", "00100", "00011"),
}


def select_evenly(items: list, count: int) -> list:
    """Keep the first, the last, and evenly spaced items between them."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[-1]]
    last = len(items) - 1
    chosen: list = []
    seen: set[int] = set()
    for step in range(count):
        index = round(step * last / (count - 1))
        if index not in seen:
            seen.add(index)
            chosen.append(items[index])
    return chosen


def snapshot_label(iteration: int) -> str:
    return f"iter {iteration:04d}"


def draw_label(image: np.ndarray, text: str, scale: int = 2) -> np.ndarray:
    """Paint a dark strip and an ASCII label along the top of an RGB frame."""
    frame = np.array(image, copy=True)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected an RGB frame, got shape {frame.shape}")
    band = 7 * scale + 10
    frame[:band, :, :] = (18, 18, 18)
    cursor_x = 8
    cursor_y = 5
    color = np.array((255, 255, 255), dtype=frame.dtype)
    height, width = frame.shape[:2]
    for character in text:
        glyph = _GLYPHS.get(character, _GLYPHS[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit != "1":
                    continue
                y0 = cursor_y + row * scale
                x0 = cursor_x + col * scale
                y1 = min(height, y0 + scale)
                x1 = min(width, x0 + scale)
                if y0 < height and x0 < width:
                    frame[y0:y1, x0:x1] = color
        cursor_x += (5 + 1) * scale
    return frame


def tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Place frames left to right, then top to bottom. Two columns when there are two or more."""
    if not frames:
        raise ValueError("no frames to tile")
    height, width = frames[0].shape[:2]
    for frame in frames:
        if frame.shape[:2] != (height, width) or frame.shape[-1] != 3:
            raise ValueError("every frame must be the same RGB size")
    columns = 1 if len(frames) == 1 else 2
    rows = (len(frames) + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        y = row * height
        x = column * width
        canvas[y : y + height, x : x + width] = frame
    return canvas


def even_size(image: np.ndarray) -> np.ndarray:
    """Pad an RGB image so both sides are even, which H.264 requires."""
    height, width = image.shape[:2]
    pad_y = height % 2
    pad_x = width % 2
    if pad_y == 0 and pad_x == 0:
        return image
    out = np.zeros((height + pad_y, width + pad_x, 3), dtype=np.uint8)
    out[:height, :width] = image
    return out
