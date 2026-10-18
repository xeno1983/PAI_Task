"""Terminal color helpers (no simulator)."""

from __future__ import annotations

import unittest

from pick_and_carry.console import delta_better, format_metric, paint


class ConsoleTests(unittest.TestCase):
    def test_delta_higher_is_better(self):
        self.assertTrue(delta_better(0.2, 0.1, higher_is_better=True))
        self.assertFalse(delta_better(0.1, 0.2, higher_is_better=True))
        self.assertIsNone(delta_better(0.1, 0.1, higher_is_better=True))
        self.assertIsNone(delta_better(0.1, None, higher_is_better=True))

    def test_delta_lower_is_better(self):
        self.assertTrue(delta_better(0.1, 0.2, higher_is_better=False))
        self.assertFalse(delta_better(0.2, 0.1, higher_is_better=False))

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
