"""Unit tests for in-place Genesis FPS logging (no simulator)."""

from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest import mock


class GenesisFpsTests(unittest.TestCase):
    def test_write_inplace_rewrites_same_line(self):
        from pick_and_carry import genesis_fps as mod

        stream = io.StringIO()
        formatter = mock.Mock()
        formatter.format.return_value = "Running at 100.00 FPS."
        lock = mock.MagicMock()
        lock.__enter__ = mock.Mock(return_value=None)
        lock.__exit__ = mock.Mock(return_value=False)
        logger = SimpleNamespace(
            _stream=stream,
            _formatter=formatter,
            _is_new_line=True,
            lock_timer=mock.Mock(return_value=lock),
        )
        mod._last_width = 0
        mod._write_inplace(logger, "Running at ~<100.00>~ FPS.")
        self.assertTrue(stream.getvalue().startswith("\r"))
        self.assertFalse(logger._is_new_line)
        first = stream.getvalue()
        formatter.format.return_value = "Running at 200.00 FPS."
        mod._write_inplace(logger, "Running at ~<200.00>~ FPS.")
        self.assertEqual(stream.getvalue().count("\r"), 2)
        self.assertGreater(len(stream.getvalue()), len(first))

    def test_end_fps_status_line_adds_newline(self):
        from pick_and_carry import genesis_fps as mod

        stream = io.StringIO()
        logger = SimpleNamespace(_stream=stream, _is_new_line=False)
        with mock.patch.dict("sys.modules", {"genesis": SimpleNamespace(logger=logger)}):
            # genesis may already be imported; patch gs.logger via module attribute.
            import genesis as gs

            old = gs.logger
            gs.logger = logger
            try:
                mod.end_fps_status_line()
            finally:
                gs.logger = old
        self.assertTrue(stream.getvalue().endswith("\n"))
        self.assertTrue(logger._is_new_line)


if __name__ == "__main__":
    unittest.main()
