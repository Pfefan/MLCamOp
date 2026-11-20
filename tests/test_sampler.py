import unittest

import numpy as np

from src.data.sampler import (
    SampleSet, split_dataset, concert_split, val_concert_indices, balance_concert_weights,
)


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


class TestValConcertIndices(unittest.TestCase):

    def setUp(self):
        self.data_cfg = {"concerts": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}

    def test_held_out_by_id_regardless_of_position(self):
        self.data_cfg["val_concerts"] = ["c", "a"]
        self.assertEqual(val_concert_indices(self.data_cfg), [0, 2])

    def test_defaults_to_last_concert(self):
        self.assertEqual(val_concert_indices(self.data_cfg), [2])

    def test_unknown_id_is_rejected(self):
        self.data_cfg["val_concerts"] = ["b", "typo"]
        with self.assertRaises(ValueError):
            val_concert_indices(self.data_cfg)

    def test_holding_out_every_concert_is_rejected(self):
        self.data_cfg["val_concerts"] = ["a", "b", "c"]
        with self.assertRaises(ValueError):
            val_concert_indices(self.data_cfg)


class TestBalanceConcertWeights(unittest.TestCase):

    def _pair_mass(self, concert: SampleSet, cls: int) -> float:
        return sum(w for w, l in zip(concert.weights, concert.labels) if l == cls)

    def test_every_concert_class_pair_gets_equal_mass(self):
        skewed = SampleSet(np.zeros((10, 1)), [0] * 8 + [1] * 2, [1.0] * 10)
        long   = SampleSet(np.zeros((30, 1)), [0] * 15 + [1] * 15, [1.0] * 30)

        balanced = balance_concert_weights([skewed, long])

        masses = [self._pair_mass(c, cls) for c in balanced for cls in (0, 1)]
        for mass in masses:
            self.assertAlmostEqual(mass, 10.0)
        self.assertAlmostEqual(sum(masses) / 40, 1.0)
        self.assertEqual(skewed.weights, [1.0] * 10)

    def test_transition_downweighting_is_kept_within_a_pair(self):
        concert = SampleSet(np.zeros((4, 1)), [0, 0, 1, 1], [1.0, 0.3, 1.0, 1.0])

        weights = balance_concert_weights([concert])[0].weights

        self.assertAlmostEqual(weights[1] / weights[0], 0.3)
        self.assertAlmostEqual(weights[0] + weights[1], weights[2] + weights[3])


if __name__ == "__main__":
    unittest.main()
