import unittest

import numpy as np

from src.data.sampler import SampleSet, split_dataset, concert_split


def _make_sampleset(n: int, fill: int = 0) -> SampleSet:
    frames  = np.full((n, 2, 2, 3), fill, dtype=np.uint8)
    labels  = [i % 2 for i in range(n)]
    weights = [1.0] * n
    return SampleSet(frames, labels, weights)


class TestSplitDataset(unittest.TestCase):

    def test_split_sizes(self):
        data = _make_sampleset(100)
        train, val = split_dataset(data, val_split=0.2)
        self.assertEqual(len(train.frames), 80)
        self.assertEqual(len(val.frames), 20)
        self.assertEqual(len(train.labels), 80)
        self.assertEqual(len(val.labels), 20)
        self.assertEqual(len(train.weights), 80)

    def test_deterministic_with_seed(self):
        data = SampleSet(np.zeros((50, 2, 2, 3), dtype=np.uint8), list(range(50)), [1.0] * 50)
        _, v1 = split_dataset(data, seed=42)
        _, v2 = split_dataset(data, seed=42)
        self.assertEqual(v1.labels, v2.labels)


class TestConcertSplit(unittest.TestCase):

    def test_splits_by_concert(self):
        c1 = _make_sampleset(10, fill=0)
        c2 = _make_sampleset(5, fill=1)
        train, val = concert_split([c1, c2], val_concert_idx=-1)
        self.assertEqual(len(train.frames), 10)
        self.assertEqual(len(val.frames), 5)

    def test_requires_two_concerts(self):
        with self.assertRaises(ValueError):
            concert_split([_make_sampleset(1)], val_concert_idx=0)


if __name__ == "__main__":
    unittest.main()
