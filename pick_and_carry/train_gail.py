"""Compatibility entry point: GAIL menu / CLI now runs ValueDICE.

Prefer::

    python -m pick_and_carry.train_valuedice --easy-walk ...

This module re-exports resume helpers for older tests and forwards ``main``.
"""

from __future__ import annotations

from pick_and_carry.train_valuedice import (  # noqa: F401
    _gail_iteration,
    _resolve_resume_path,
    _valuedice_iteration,
    main,
    parse_args,
    run_valuedice,
)

# Legacy name used in docs / mental model.
run_gail = run_valuedice


if __name__ == "__main__":
    main()
