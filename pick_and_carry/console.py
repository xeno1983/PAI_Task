"""ANSI terminal helpers (Windows VT enabled when possible)."""

from __future__ import annotations

import os
import sys

_RESET = "\033[0m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[90m"

_ansi_ready: bool | None = None


def _enable_windows_vt() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def color_enabled() -> bool:
    global _ansi_ready
    if _ansi_ready is not None:
        return _ansi_ready
    if os.environ.get("NO_COLOR"):
        _ansi_ready = False
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        _ansi_ready = False
        return False
    _enable_windows_vt()
    _ansi_ready = True
    return True


def paint(text: str, *, better: bool | None) -> str:
    """Green if better, red if worse, dim if unchanged / unknown."""
    if better is None or not color_enabled():
        return text
    return f"{_GREEN if better else _RED}{text}{_RESET}"


def delta_better(current: float, previous: float | None, *, higher_is_better: bool, eps: float = 1e-9) -> bool | None:
    if previous is None:
        return None
    diff = current - previous
    if abs(diff) <= eps:
        return None
    return (diff > 0) if higher_is_better else (diff < 0)


def abs_delta_better(current: float, previous: float | None, *, eps: float = 1e-9) -> bool | None:
    """True when ``|current|`` shrank (useful for signed losses)."""
    if previous is None:
        return None
    diff = abs(current) - abs(previous)
    if abs(diff) <= eps:
        return None
    return diff < 0


def toward_target_better(
    current: float,
    previous: float | None,
    *,
    target: float,
    eps: float = 1e-9,
) -> bool | None:
    """True when ``current`` moved closer to ``target``."""
    if previous is None:
        return None
    d_now = abs(current - target)
    d_prev = abs(previous - target)
    if abs(d_now - d_prev) <= eps:
        return None
    return d_now < d_prev


def format_metric(
    name: str,
    value: float,
    previous: float | None,
    *,
    higher_is_better: bool,
    fmt: str,
) -> str:
    body = f"{name} {value:{fmt}}"
    return paint(body, better=delta_better(value, previous, higher_is_better=higher_is_better))


def format_abs_metric(name: str, value: float, previous: float | None, *, fmt: str) -> str:
    body = f"{name} {value:{fmt}}"
    return paint(body, better=abs_delta_better(value, previous))


def format_toward_metric(
    name: str,
    value: float,
    previous: float | None,
    *,
    target: float,
    fmt: str,
) -> str:
    body = f"{name} {value:{fmt}}"
    return paint(body, better=toward_target_better(value, previous, target=target))
