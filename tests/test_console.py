"""Terminal color helpers (no simulator)."""

from __future__ import annotations

import unittest

from pick_and_carry.console import (
    abs_delta_better,
    delta_better,
    format_metric,
    paint,
    toward_target_better,
)


class ConsoleTests(unittest.TestCase):
    def test_delta_higher_is_better(self):
        self.assertTrue(delta_better(0.2, 0.1, higher_is_better=True))
        self.assertFalse(delta_better(0.1, 0.2, higher_is_better=True))
        self.assertIsNone(delta_better(0.1, 0.1, higher_is_better=True))
        self.assertIsNone(delta_better(0.1, None, higher_is_better=True))

    def test_delta_lower_is_better(self):
        self.assertTrue(delta_better(0.1, 0.2, higher_is_better=False))
        self.assertFalse(delta_better(0.2, 0.1, higher_is_better=False))

    def test_abs_and_toward_helpers(self):
        self.assertTrue(abs_delta_better(-0.01, -0.05))
        self.assertFalse(abs_delta_better(-0.05, -0.01))
        self.assertTrue(toward_target_better(0.70, 0.95, target=0.65))
        self.assertFalse(toward_target_better(0.95, 0.70, target=0.65))

    def test_format_without_color_when_disabled(self):
        import pick_and_carry.console as console

        old = console._ansi_ready
        try:
            console._ansi_ready = False
            text = format_metric("r_env", 0.02, 0.01, higher_is_better=True, fmt=".3f")
            self.assertEqual(text, "r_env 0.020")
            self.assertEqual(paint("x", better=True), "x")
        finally:
            console._ansi_ready = old


if __name__ == "__main__":
    unittest.main()
