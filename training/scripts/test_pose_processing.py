"""Focused tests for temporal gaps, confidence masks, geometry and source splits."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dance_core.pose import normalise_window, resample_pose, window_starts
from prepare_stgcn import aist_assignments, brace_assignments


class ProcessingTests(unittest.TestCase):
    def moving_pose(self, times):
        points = np.ones((len(times), 17, 3), dtype=float)
        points[..., 0] = 100 + np.arange(17)[None, :] * 10 + np.asarray(times)[:, None] * 20
        points[..., 1] = 100 + np.arange(17)[None, :] * 12
        return points

    def test_different_fps_preserves_physical_time(self):
        for fps in (25, 29.97003, 60):
            times = np.arange(int(fps * 3)) / fps
            points, observed, grid = resample_pose(self.moving_pose(times), times, fps=30)
            np.testing.assert_allclose(np.diff(grid), 1 / 30)
            np.testing.assert_allclose(points[:, 0, 0], 100 + grid * 20, atol=1e-4)
            self.assertTrue(observed.all())
            self.assertAlmostEqual(grid[-1], times[-1], delta=1 / 30)

    def test_short_missing_gap_filled_long_gap_remains_zero(self):
        times = np.arange(90) / 30
        raw = self.moving_pose(times)
        raw[10:12, 8] = np.nan
        raw[30:48, 8] = np.nan
        points, observed, _ = resample_pose(raw, times)
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue((points[10:12, 8, 2] > 0).all())
        self.assertFalse(observed[10:12, 8].any())
        self.assertTrue((points[30:48, 8] == 0).all())

    def test_no_extrapolation_of_invisible_joints(self):
        times = np.arange(60) / 30
        raw = self.moving_pose(times)
        raw[:10, 9] = 0
        raw[-10:, 9] = 0
        points, _, _ = resample_pose(raw, times)
        self.assertTrue((points[:10, 9] == 0).all())
        self.assertTrue((points[-10:, 9] == 0).all())

    def test_window_boundaries_never_pad_or_wrap(self):
        self.assertEqual(list(window_starts(59, 60, 30)), [])
        self.assertEqual(list(window_starts(60, 60, 30)), [0])
        self.assertEqual(list(window_starts(121, 60, 30)), [0, 30, 60])

    def test_normalisation_keeps_geometry_and_missing_mask(self):
        times = np.arange(60) / 30
        raw = self.moving_pose(times)
        raw[:, 0] = 0
        observed = raw[..., 2] > 0
        normal, _, reason = normalise_window(raw, observed)
        self.assertIsNone(reason)
        self.assertTrue((normal[:, 0] == 0).all())
        before = raw[0, 12, :2] - raw[0, 5, :2]
        after = normal[0, 12, :2] - normal[0, 5, :2]
        self.assertAlmostEqual(float(before[0] / before[1]), float(after[0] / after[1]), places=5)
        upside_down = raw.copy()
        upside_down[:, 1:, 1] = 500 - upside_down[:, 1:, 1]
        inverted, _, _ = normalise_window(upside_down, observed)
        self.assertGreater(normal[0, 12, 1] - normal[0, 5, 1], 0)
        self.assertLess(inverted[0, 12, 1] - inverted[0, 5, 1], 0)

    def test_low_visibility_is_rejected(self):
        raw = self.moving_pose(np.arange(60) / 30)
        raw[:, 5:14] = 0
        result, _, reason = normalise_window(raw, raw[..., 2] > 0)
        self.assertIsNone(result)
        self.assertIsNotNone(reason)

    def test_source_splits_are_deterministic_and_grouped(self):
        rows = [dict(genre='test', dancer=str(dancer), sequence_id='%s_%s' % (dancer, i))
                for dancer in range(3) for i in range(5)]
        assignment, dancers = aist_assignments(rows, 42, 0)
        self.assertEqual(assignment, aist_assignments(rows, 42, 0)[0])
        self.assertEqual(len({d for values in dancers.values() for d in values}), 3)
        for dancer in range(3):
            self.assertEqual(len({assignment[r['sequence_id']] for r in rows if r['dancer'] == str(dancer)}), 1)
        videos = [dict(video_id=str(v), action=a) for v in range(15)
                  for a in ('toprock', 'footwork', 'powermove')]
        assignment = brace_assignments(videos, 42)
        self.assertEqual(assignment, brace_assignments(videos, 42))
        self.assertEqual(set(assignment.values()), {'train', 'val', 'test'})


if __name__ == '__main__':
    unittest.main()
