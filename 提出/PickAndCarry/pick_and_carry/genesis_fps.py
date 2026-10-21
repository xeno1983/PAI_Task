"""Rewrite Genesis FPS status on one terminal line instead of spamming.

Genesis ``FPSTracker`` calls ``gs.logger.info`` about every 50 ms. This module
patches that path to use carriage-return updates, and exposes a helper so other
``print`` calls can finish the line cleanly first.
"""

from __future__ import annotations

import logging
from typing import Any

_PATCHED = False
_last_width = 0


def end_fps_status_line() -> None:
    """Finish an in-place FPS line so the next ``print`` starts on a new row."""
    try:
        import genesis as gs
    except Exception:
        return
    logger = getattr(gs, "logger", None)
    if logger is None:
        return
    stream = getattr(logger, "_stream", None)
    if stream is None or getattr(stream, "closed", True):
        return
    if not getattr(logger, "_is_new_line", True):
        stream.write("\n")
        stream.flush()
        logger._is_new_line = True


def _write_inplace(logger: Any, message: str) -> None:
    global _last_width
    stream = logger._stream
    if stream is None or getattr(stream, "closed", True):
        return
    record = logging.LogRecord(
        name="genesis",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    formatted = logger._formatter.format(record)
    width = max(_last_width, len(formatted))
    _last_width = width
    with logger.lock_timer():
        stream.write("\r" + formatted + (" " * (width - len(formatted))))
        stream.flush()
        logger._is_new_line = False


def install_inplace_fps_logging() -> None:
    """Patch ``FPSTracker.step`` so ``Running at … FPS`` rewrites one line."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from genesis.utils.tools import FPSTracker
    except Exception:
        return

    original_step = FPSTracker.step

    def step(self, count: bool = True, current_time: float | None = None):
        import genesis as gs

        logger = getattr(gs, "logger", None)
        if logger is None or not getattr(self, "log", True):
            return original_step(self, count=count, current_time=current_time)

        real_info = logger.info

        def info_hook(message, *args, **kwargs):
            text = str(message)
            if "Running at" in text and "FPS" in text:
                _write_inplace(logger, text)
                return
            return real_info(message, *args, **kwargs)

        logger.info = info_hook
        try:
            return original_step(self, count=count, current_time=current_time)
        finally:
            logger.info = real_info

    FPSTracker.step = step  # type: ignore[method-assign]

    # ``print`` does not know about the half-written FPS line; finish it first.
    import builtins

    if not getattr(builtins.print, "_genesis_fps_wrapped", False):
        real_print = builtins.print

        def print_hook(*args, **kwargs):
            end_fps_status_line()
            return real_print(*args, **kwargs)

        print_hook._genesis_fps_wrapped = True  # type: ignore[attr-defined]
        builtins.print = print_hook  # type: ignore[assignment]

    _PATCHED = True
