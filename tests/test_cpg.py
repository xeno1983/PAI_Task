"""Unit tests for the learnable leg CPG (no Genesis)."""

from __future__ import annotations

import math
import unittest

import torch

from pick_and_carry.config import CPG_PARAM_DIM, NUM_ACTIONS
from pick_and_carry.cpg import LegCPG, command_to_cpg_params, decode_cpg_params, encode_cpg_params


class CpgTests(unittest.TestCase):
    def test_action_layout_uses_cpg(self):
        self.assertEqual(CPG_PARAM_DIM, 10)
        self.assertEqual(NUM_ACTIONS, CPG_PARAM_DIM + 6 + 1)

    def test_encode_decode_roundtrip_amp(self):
        raw = encode_cpg_params(freq_hz=2.5, amp=0.8, reverse=False, n=1)
        decoded = decode_cpg_params(raw)
        self.assertAlmostEqual(float(decoded["amp"][0]), 0.8, places=2)
        self.assertEqual(float(decoded["going_back"][0]), 0.0)
        raw_b = encode_cpg_params(amp=0.7, reverse=True, n=1)
        decoded_b = decode_cpg_params(raw_b)
        self.assertAlmostEqual(float(decoded_b["amp"][0]), 0.7, places=2)
        self.assertEqual(float(decoded_b["going_back"][0]), 1.0)

    def test_step_stand_near_zero(self):
        cpg = LegCPG(2, device=torch.device("cpu"))
        raw = encode_cpg_params(amp=0.0, n=2)
        leg = cpg.step(raw)
        self.assertEqual(tuple(leg.shape), (2, 12))
        self.assertLess(float(leg.abs().mean()), 0.05)

    def test_step_forward_moves_thigh(self):
        cpg = LegCPG(1, device=torch.device("cpu"))
        raw = command_to_cpg_params(torch.tensor([[0.35, 0.0, 0.0]]))
        cpg.phase.fill_(math.pi / 2)
        leg = cpg.step(raw)
        self.assertGreater(float(leg[0, 1].abs()), 0.05)

    def test_command_forward_vs_backward_amp_sign(self):
        fwd = command_to_cpg_params(torch.tensor([[0.35, 0.0, 0.0]]))
        bwd = command_to_cpg_params(torch.tensor([[-0.30, 0.0, 0.0]]))
        self.assertGreater(float(fwd[0, 1]), 0.0)
        self.assertLess(float(bwd[0, 1]), 0.0)

    def test_forward_thigh_and_calf_swing_opposite(self):
        """Matching thigh/calf signs walk aft on Go2; opposite signs go +X."""
        cpg = LegCPG(1, device=torch.device("cpu"))
        raw = command_to_cpg_params(torch.tensor([[0.35, 0.0, 0.0]]))
        cpg.phase.fill_(math.pi / 2)
        leg = cpg.step(raw)
        thigh = float(leg[0, 1])
        calf = float(leg[0, 2])
        self.assertGreater(abs(thigh), 0.05)
        self.assertGreater(abs(calf), 0.05)
        self.assertLess(thigh * calf, 0.0)


if __name__ == "__main__":
    unittest.main()
