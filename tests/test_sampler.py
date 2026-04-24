import unittest

import numpy as np

from src.data.sampler import split_dataset, concert_split


class TestSplitDataset(unittest.TestCase):

    def test_split_sizes(self):
        frames = np.zeros((100, 2, 2, 3), dtype=np.uint8)
        labels = [0] * 50 + [1] * 50
        train_f, train_l, val_f, val_l = split_dataset(frames, labels, val_split=0.2)
        self.assertEqual(len(train_f), 80)
        self.assertEqual(len(val_f), 20)
        self.assertEqual(len(train_l), 80)
        self.assertEqual(len(val_l), 20)

    def test_deterministic_with_seed(self):
        frames = np.zeros((50, 2, 2, 3), dtype=np.uint8)
        labels = list(range(50))
        _, l1, _, _ = split_dataset(frames, labels, seed=42)
        _, l2, _, _ = split_dataset(frames, labels, seed=42)
        self.assertEqual(l1, l2)


class TestConcertSplit(unittest.TestCase):

    def test_splits_by_concert(self):
        c1_frames = np.zeros((10, 2, 2, 3), dtype=np.uint8)
        c2_frames = np.ones((5, 2, 2, 3), dtype=np.uint8)
        c1_labels = [0] * 10
        c2_labels = [1] * 5
        train_f, train_l, val_f, val_l = concert_split(
            [c1_frames, c2_frames], [c1_labels, c2_labels], val_concert_idx=-1
        )
        self.assertEqual(len(train_f), 10)
        self.assertEqual(len(val_f), 5)

    def test_requires_two_concerts(self):
        with self.assertRaises(ValueError):
            concert_split([np.zeros((1, 2, 2, 3), dtype=np.uint8)], [[0]], val_concert_idx=0)


if __name__ == "__main__":
    unittest.main()
