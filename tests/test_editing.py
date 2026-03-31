import unittest

import numpy as np

from src.pipeline.editing import assemble_cut, _enforce_min_shot_length


class TestEnforceMinShotLength(unittest.TestCase):

    def test_no_change_when_all_runs_long_enough(self):
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        result = _enforce_min_shot_length(labels, min_frames=3)
        self.assertEqual(result, labels)

    def test_short_run_is_merged(self):
        labels = [0, 0, 0, 1, 0, 0, 0]  # single '1' is short
        result = _enforce_min_shot_length(labels, min_frames=3)
        self.assertEqual(result, [0, 0, 0, 0, 0, 0, 0])

    def test_empty_input(self):
        self.assertEqual(_enforce_min_shot_length([], min_frames=5), [])

    def test_min_frames_one_returns_original(self):
        labels = [0, 1, 0, 1]
        self.assertEqual(_enforce_min_shot_length(labels, min_frames=1), labels)


class TestAssembleCut(unittest.TestCase):

    def test_selects_correct_source(self):
        wide = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
        close = [np.ones((2, 2, 3), dtype=np.uint8) * 255 for _ in range(3)]
        labels = [0, 1, 0]
        result = assemble_cut(wide, close, labels, min_shot_frames=1)
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], wide[0])
        np.testing.assert_array_equal(result[1], close[1])
        np.testing.assert_array_equal(result[2], wide[2])


if __name__ == "__main__":
    unittest.main()
