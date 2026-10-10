"""Snapshot selection and comparison-frame layout. No simulator."""

from __future__ import annotations

import unittest

import numpy as np

from pick_and_carry.compare import draw_label, even_size, select_evenly, snapshot_label, tile_frames


class SelectTests(unittest.TestCase):
    def test_keeps_a_short_list(self):
        self.assertEqual(select_evenly(["a", "b"], 4), ["a", "b"])

    def test_spreads_from_first_to_last(self):
        items = [f"{index:04d}" for index in range(10)]
        self.assertEqual(select_evenly(items, 4), ["0000", "0003", "0006", "0009"])

    def test_single_panel_is_the_latest(self):
        self.assertEqual(select_evenly(["early", "late"], 1), ["late"])


class FrameTests(unittest.TestCase):
    def test_label_paints_the_top_strip(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        labeled = draw_label(image, snapshot_label(20))
        self.assertEqual(labeled.shape, image.shape)
        self.assertGreater(int(labeled[:20].max()), 0)
        self.assertTrue(np.array_equal(image, np.zeros_like(image)))

    def test_two_frames_sit_side_by_side(self):
        left = np.full((4, 6, 3), 10, dtype=np.uint8)
        right = np.full((4, 6, 3), 20, dtype=np.uint8)
        tiled = tile_frames([left, right])
        self.assertEqual(tiled.shape, (4, 12, 3))
        self.assertEqual(int(tiled[0, 0, 0]), 10)
        self.assertEqual(int(tiled[0, 6, 0]), 20)

    def test_even_pad(self):
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        padded = even_size(image)
        self.assertEqual(padded.shape, (4, 6, 3))
