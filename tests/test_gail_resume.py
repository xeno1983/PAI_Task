"""GAIL resume helpers (no simulator)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch


class GailResumeTests(unittest.TestCase):
    def test_resolve_latest_uses_output(self):
        from pick_and_carry.train_gail import _resolve_resume_path

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            out = log_dir / "gail_trot.pt"
            out.write_bytes(b"x")
            self.assertEqual(_resolve_resume_path("latest", log_dir, out), out)

    def test_gail_iteration_prefers_explicit_field(self):
        from pick_and_carry.train_gail import _gail_iteration

        self.assertEqual(_gail_iteration({"gail_iteration": 300, "iteration": 0}), 300)
        self.assertEqual(_gail_iteration({"iteration": 0}), 0)

    def test_parse_resume_and_bc_default(self):
        from pick_and_carry.train_gail import parse_args
        import sys

        old = sys.argv
        try:
            sys.argv = ["train_gail", "--resume", "latest"]
            args = parse_args()
            self.assertEqual(args.resume, "latest")
            self.assertIsNone(args.bc_epochs)
            self.assertEqual(args.iterations, 1000)
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
