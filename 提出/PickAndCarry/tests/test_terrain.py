"""Rough-course height field. No simulator."""

from __future__ import annotations

import unittest

import numpy as np

from pick_and_carry.terrain import (
    GOAL_FLAT,
    GOAL_X_RANGE,
    OBJECT_X_RANGE,
    PICKUP_FLAT,
    ROUGH_1,
    ROUGH_2,
    make_course,
    make_supply_yard,
    make_terrain_bank,
    paste_tile,
    rough_mask,
    sample_height,
)


class CourseTests(unittest.TestCase):
    def test_pads_are_flat_and_strips_are_bumpy(self):
        height, origin, scale = make_course(amplitude=0.035, horizontal_scale=0.05)
        self.assertGreater(float(height.max()), 0.02)
        self.assertLessEqual(float(height.max()), 0.035 + 1e-5)
        self.assertGreaterEqual(float(height.min()), 0.0)

        def at(x: float, y: float = 0.0) -> float:
            return float(sample_height(height, origin, scale, np.array([[x, y]], dtype=np.float32))[0])

        self.assertAlmostEqual(at(0.0), 0.0, places=4)
        self.assertAlmostEqual(at(0.5 * (PICKUP_FLAT[0] + PICKUP_FLAT[1])), 0.0, places=4)
        self.assertAlmostEqual(at(0.5 * (GOAL_FLAT[0] + GOAL_FLAT[1])), 0.0, places=4)
        for x0, x1 in (ROUGH_1, ROUGH_2):
            xs = np.linspace(x0, x1, 30)
            samples = sample_height(height, origin, scale, np.stack((xs, np.zeros_like(xs)), axis=1))
            self.assertGreater(float(samples.max()), 0.02)

    def test_steps_are_flat_treads_between_the_pads(self):
        height, origin, scale = make_course(amplitude=0.045, horizontal_scale=0.05, steps=True)

        def at(x: float) -> float:
            return float(sample_height(height, origin, scale, np.array([[x, 0.0]], dtype=np.float32))[0])

        self.assertAlmostEqual(at(1.6), 0.0, places=4)
        self.assertAlmostEqual(at(4.0), 0.0, places=4)
        self.assertGreater(at(0.70), 0.04)
        self.assertGreater(at(2.56), 0.04)
        self.assertGreater(at(2.90), 0.04)
        self.assertLess(at(2.74), at(2.56))
        self.assertLessEqual(float(height.max()), 0.045 + 1e-5)

    def test_each_draw_changes_the_bumps_and_keeps_the_pads(self):
        rng = np.random.default_rng(0)
        first, origin, scale = make_course(amplitude=0.035, horizontal_scale=0.05, rng=rng)
        second, _, _ = make_course(amplitude=0.035, horizontal_scale=0.05, rng=rng)
        self.assertFalse(np.allclose(first, second))
        self.assertLessEqual(float(second.max()), 0.035 + 1e-5)
        pad_x = 0.5 * (PICKUP_FLAT[0] + PICKUP_FLAT[1])
        height = float(sample_height(second, origin, scale, np.array([[pad_x, 0.0]], dtype=np.float32))[0])
        self.assertAlmostEqual(height, 0.0, places=4)

    def test_replacing_one_tile_leaves_the_other(self):
        rng = np.random.default_rng(1)
        height, origin, scale, shifts, cols, stride = make_terrain_bank(2, rng=rng)
        before = height.copy()
        replacement, _, _ = make_course(amplitude=0.035, horizontal_scale=scale, rng=rng)
        paste_tile(height, 1, replacement, cols, stride)
        self.assertTrue(np.array_equal(height[:, :cols], before[:, :cols]))
        self.assertFalse(np.array_equal(height[:, stride : stride + cols], before[:, stride : stride + cols]))
        point = np.array([[1.7, float(shifts[1, 1])]], dtype=np.float32)
        sampled = float(sample_height(height, origin, scale, point)[0])
        local = float(sample_height(replacement, origin, scale, np.array([[1.7, 0.0]], dtype=np.float32))[0])
        self.assertAlmostEqual(sampled, local, places=5)

    def test_supply_yard_pads_are_flat_and_wide(self):
        height, origin, scale, pickups, goals = make_supply_yard(4, amplitude=0.05)
        self.assertGreater(origin[1] + (height.shape[1] - 1) * scale - origin[1], 5.0)
        self.assertGreater(float(np.std(pickups[:, 1])), 0.4)
        for point in (*pickups, *goals):
            sample = float(sample_height(height, origin, scale, np.array([point], dtype=np.float32))[0])
            self.assertAlmostEqual(sample, 0.0, places=4)
        self.assertGreater(float(height.max()), 0.02)

    def test_spawns_sit_on_flat_pads(self):
        for x in OBJECT_X_RANGE:
            self.assertEqual(float(rough_mask(np.array([x]))[0]), 0.0)
        for x in GOAL_X_RANGE:
            self.assertEqual(float(rough_mask(np.array([x]))[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
