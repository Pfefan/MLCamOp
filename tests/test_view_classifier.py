import unittest

import numpy as np
import torch

from src.data.sampler import SampleSet
from src.models.view_classifier import (
    ViewClassifier, _preprocess_frame, _FrameDataset, _TEMPORAL_WINDOW,
    _augment_batch, _validation_scores,
)


class TestPreprocessFrame(unittest.TestCase):

    def test_output_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = _preprocess_frame(frame)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_output_dtype(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = _preprocess_frame(frame)
        self.assertTrue(tensor.dtype.is_floating_point)

    def test_dual_frame_output_shape(self):
        frame = np.random.randint(0, 255, (224, 224, 6), dtype=np.uint8)
        tensor = _preprocess_frame(frame)
        self.assertEqual(tensor.shape, (6, 224, 224))


class TestFrameDataset(unittest.TestCase):

    def setUp(self):
        # Use 6-channel dual frames (the default mode) with enough frames for windows
        n = max(10, _TEMPORAL_WINDOW + 5)
        self.frames = [np.random.randint(0, 255, (224, 224, 6), dtype=np.uint8) for _ in range(n)]
        self.labels = [i % 2 for i in range(n)]

    def test_length(self):
        ds = _FrameDataset(self.frames, self.labels)
        self.assertEqual(len(ds), len(self.frames))

    def test_getitem_returns_tensor_and_label(self):
        ds = _FrameDataset(self.frames, self.labels)
        tensor, label = ds[0]
        self.assertEqual(tensor.shape, (_TEMPORAL_WINDOW * 6, 224, 224))
        self.assertEqual(label, 0)

    def test_augmented_batch_same_shape(self):
        ds = _FrameDataset(self.frames, self.labels)
        batch = torch.stack([ds[0][0], ds[1][0]])
        augmented = _augment_batch(batch)
        self.assertEqual(augmented.shape, (2, _TEMPORAL_WINDOW * 6, 224, 224))
        self.assertEqual(augmented.dtype, torch.uint8)


class TestViewClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = ViewClassifier()

    def test_predict_returns_int(self):
        frame = np.random.randint(0, 255, (224, 224, 6), dtype=np.uint8)
        pred = self.classifier.predict(frame)
        self.assertIn(pred, (0, 1))

    def test_predict_batch(self):
        frames = [np.random.randint(0, 255, (224, 224, 6), dtype=np.uint8) for _ in range(5)]
        preds = self.classifier.predict_batch(frames)
        self.assertEqual(len(preds), 5)
        for p in preds:
            self.assertIn(p, (0, 1))

    def test_dual_frame_predict(self):
        classifier = ViewClassifier()
        frame = np.random.randint(0, 255, (224, 224, 6), dtype=np.uint8)
        pred = classifier.predict(frame)
        self.assertIn(pred, (0, 1))


def _random_concert(n: int) -> SampleSet:
    frames = np.random.randint(0, 255, (n, 6, 224, 224), dtype=np.uint8)
    return SampleSet(frames, [i % 2 for i in range(n)], [1.0] * n)


class TestTraining(unittest.TestCase):

    def test_frozen_blocks_stay_in_eval_mode_while_training(self):
        model = ViewClassifier().model
        unfrozen = model._UNFREEZE_BLOCKS

        model.train()

        for block in model.features[:-unfrozen]:
            self.assertFalse(any(module.training for module in block.modules()))
        for block in model.features[-unfrozen:]:
            self.assertTrue(all(module.training for module in block.modules()))

    def test_validation_scores_weigh_concerts_equally(self):
        short_labels, long_labels = [0, 1], [0, 1, 0, 1, 0, 1]
        right = np.eye(2)[short_labels]
        wrong = np.eye(2)[[1 - label for label in long_labels]]

        accuracy, bal_acc, _ = _validation_scores(
            np.concatenate([right, wrong]), [short_labels, long_labels],
            smooth_penalty=0.0, smooth_min_shot=1, cut_tolerance_samples=1,
        )

        self.assertAlmostEqual(accuracy, 2 / 8)
        self.assertAlmostEqual(bal_acc, 0.5)

    def test_trains_against_multiple_val_concerts(self):
        classifier = ViewClassifier()
        classifier.train(
            train_concerts=[_random_concert(8), _random_concert(6)],
            val_concerts=[_random_concert(4), _random_concert(5)],
            epochs=1, batch_size=4, num_workers=0, use_compile=False,
        )
        self.assertTrue(classifier.is_trained)


if __name__ == "__main__":
    unittest.main()
