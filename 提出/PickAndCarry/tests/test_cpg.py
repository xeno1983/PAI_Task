"""Unit tests for the learnable leg CPG (no Genesis)."""

from __future__ import annotations

import math
import unittest

import torch

from pick_and_carry.config import CPG_PARAM_DIM, NUM_ACTIONS
from pick_and_carry.cpg import (
    LegCPG,
    command_is_moving,
    command_to_cpg_params,
    decode_cpg_params,
    encode_cpg_params,
    gate_cpg_amp_by_command,
    nominal_leg_phases,
)


class CpgTests(unittest.TestCase):
    def test_action_layout_uses_cpg(self):
        self.assertEqual(CPG_PARAM_DIM, 15)
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

    def test_encode_decode_phase_adapt_defaults(self):
        raw = encode_cpg_params(amp=0.5, couple=1.0, entrain=0.0, n=1)
        p = decode_cpg_params(raw)
        self.assertAlmostEqual(float(p["dphi_ab"][0]), 0.0, places=3)
        self.assertAlmostEqual(float(p["couple"][0]), 1.0, places=2)
        self.assertAlmostEqual(float(p["entrain"][0]), 0.0, places=2)

    def test_step_stand_near_zero(self):
        cpg = LegCPG(2, device=torch.device("cpu"))
        raw = encode_cpg_params(amp=0.0, n=2)
        leg = cpg.step(raw)
        self.assertEqual(tuple(leg.shape), (2, 12))
        self.assertLess(float(leg.abs().mean()), 0.05)

    def test_step_forward_moves_thigh(self):
        cpg = LegCPG(1, device=torch.device("cpu"))
        raw = command_to_cpg_params(torch.tensor([[0.35, 0.0, 0.0]]))
        cpg.phase = math.pi / 2
        leg = cpg.step(raw)
        self.assertGreater(float(leg[0, 1].abs()), 0.05)

    def test_command_forward_vs_backward_amp_sign(self):
        fwd = command_to_cpg_params(torch.tensor([[0.35, 0.0, 0.0]]))
        bwd = command_to_cpg_params(torch.tensor([[-0.30, 0.0, 0.0]]))
        self.assertGreater(float(fwd[0, 1]), 0.0)
        self.assertLess(float(bwd[0, 1]), 0.0)
        self.assertEqual(fwd.shape[-1], CPG_PARAM_DIM)

    def test_forward_thigh_and_calf_same_swing_sign(self):
        """Genesis-locked: same thigh/calf swing sign + sagittal=+1 walks toward +X."""
        from pick_and_carry.cpg import _CALF_SWING_SIGN, _SAGITTAL_SIGN

        cpg = LegCPG(1, device=torch.device("cpu"))
        raw = command_to_cpg_params(torch.tensor([[0.35, 0.0, 0.0]]))
        cpg.phase = math.pi / 2
        leg = cpg.step(raw)
        thigh = float(leg[0, 1])
        calf = float(leg[0, 2])
        self.assertGreater(abs(thigh), 0.05)
        self.assertGreater(abs(calf), 0.05)
        self.assertEqual(_SAGITTAL_SIGN, 1.0)
        self.assertEqual(_CALF_SWING_SIGN, 1.0)
        self.assertGreater(thigh * calf, 0.0)
        self.assertGreater(thigh, 0.0)

    def test_forward_and_backward_leg_actions_flip(self):
        """Backward (amp channel < 0) must invert calf phase relative to forward."""
        cpg_f = LegCPG(1, device=torch.device("cpu"))
        cpg_b = LegCPG(1, device=torch.device("cpu"))
        fwd = command_to_cpg_params(torch.tensor([[0.35, 0.0, 0.0]]))
        bwd = command_to_cpg_params(torch.tensor([[-0.35, 0.0, 0.0]]))
        cpg_f.phase = math.pi / 2
        cpg_b.phase = math.pi / 2
        leg_f = cpg_f.step(fwd)
        leg_b = cpg_b.step(bwd)
        self.assertAlmostEqual(float(leg_f[0, 1]), float(leg_b[0, 1]), places=3)
        self.assertLess(float(leg_f[0, 2] * leg_b[0, 2]), 0.0)

    def test_dphi_ab_shifts_pair_b(self):
        """Pair-B offset changes FR/RL phases relative to rigid π."""
        cpg0 = LegCPG(1, device=torch.device("cpu"))
        cpg1 = LegCPG(1, device=torch.device("cpu"))
        base = encode_cpg_params(amp=0.8, couple=1.0, entrain=0.0, n=1)
        shifted = encode_cpg_params(amp=0.8, dphi_ab=0.5, couple=1.0, entrain=0.0, n=1)
        cpg0.phase = 0.0
        cpg1.phase = 0.0
        # One coupling step from aligned state.
        cpg0.step(base)
        cpg1.step(shifted)
        # FR (1) and RL (2) should differ when dphi_ab ≠ 0.
        self.assertGreater(
            float((cpg0.phase_legs[0, 1] - cpg1.phase_legs[0, 1]).abs()),
            0.2,
        )

    def test_soft_couple_pulls_toward_diagonal(self):
        cpg = LegCPG(1, device=torch.device("cpu"))
        # Break diagonal: set FR equal to FL instead of FL+π.
        cpg.phase_legs[0] = torch.tensor([0.0, 0.0, math.pi, 0.0])
        raw = encode_cpg_params(amp=0.8, couple=1.0, entrain=0.0, n=1)
        cpg.step(raw)
        # After stiff coupling, FR should be near π (mod 2π).
        fr = float(cpg.phase_legs[0, 1])
        err = abs(math.atan2(math.sin(fr - math.pi), math.cos(fr - math.pi)))
        self.assertLess(err, 0.35)

    def test_contact_entrainment_pulls_toward_stance(self):
        cpg = LegCPG(1, device=torch.device("cpu"))
        raw = encode_cpg_params(amp=0.8, couple=0.2, entrain=0.8, n=1)
        # Seed edge history (post-reset warm-up); no rising pull yet.
        cpg.step(raw, foot_contact=torch.zeros(1, 4))
        cpg.phase = math.pi  # mid-swing-ish for sin-based lift
        contact = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        before = float(cpg.phase_legs[0, 0])
        cpg.step(raw, foot_contact=contact)
        after = float(cpg.phase_legs[0, 0])
        dist_before = abs(math.atan2(math.sin(before), math.cos(before)))
        dist_after = abs(math.atan2(math.sin(after), math.cos(after)))
        self.assertLess(dist_after, dist_before)

    def test_reset_suppresses_false_touchdown_entrainment(self):
        """Stance feet already down after reset must not look like rising edges."""
        cpg = LegCPG(1, device=torch.device("cpu"))
        cpg.reset()
        cpg.phase = math.pi / 2
        raw = encode_cpg_params(amp=0.8, couple=0.05, entrain=0.9, n=1)
        stance = torch.ones(1, 4)
        cpg.step(raw, foot_contact=stance)
        # First contact sample only seeds history. A buggy rising pull would yank
        # every stance leg toward 0 (wrapped distance ≪ π/2).
        for i in range(4):
            dist0 = abs(
                math.atan2(math.sin(float(cpg.phase_legs[0, i])), math.cos(float(cpg.phase_legs[0, i])))
            )
            self.assertGreater(dist0, 0.9)


    def test_nominal_leg_phases_diagonal(self):
        carrier = torch.tensor([0.0])
        z = torch.zeros(1)
        ph = nominal_leg_phases(carrier, z, z, z)
        self.assertAlmostEqual(float(ph[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(ph[0, 1]), math.pi, places=5)
        self.assertAlmostEqual(float(ph[0, 2]), math.pi, places=5)
        self.assertAlmostEqual(float(ph[0, 3]), 0.0, places=5)

    def test_gate_cpg_amp_zeros_when_command_still(self):
        params = encode_cpg_params(amp=0.9, bias_th=0.4, n=2)
        still = torch.zeros(2, 3)
        walk = torch.tensor([[0.35, 0.0, 0.0], [0.0, 0.0, 0.7]])
        gated_still = gate_cpg_amp_by_command(params, still)
        gated_walk = gate_cpg_amp_by_command(params, walk)
        self.assertFalse(bool(command_is_moving(still).any()))
        self.assertTrue(bool(command_is_moving(walk).all()))
        self.assertTrue(torch.allclose(gated_still[:, 1], torch.zeros(2)))
        self.assertTrue(torch.allclose(gated_walk[:, 1], params[:, 1]))
        # Pose biases (sit/jump) must survive the amp gate.
        self.assertTrue(torch.allclose(gated_still[:, 8], params[:, 8]))

    def test_still_command_keeps_amp_gate_closed(self):
        """Regression: after stand/jump, a walk command would re-open the amp gate."""
        params = encode_cpg_params(amp=0.9, n=1)
        still = torch.zeros(1, 3)
        self.assertTrue(torch.allclose(gate_cpg_amp_by_command(params, still)[:, 1], torch.zeros(1)))
        leaked = torch.tensor([[0.28, -0.20, 0.0]])
        self.assertTrue(bool(command_is_moving(leaked).all()))
        self.assertFalse(
            torch.allclose(gate_cpg_amp_by_command(params, leaked)[:, 1], torch.zeros(1))
        )


if __name__ == "__main__":
    unittest.main()
