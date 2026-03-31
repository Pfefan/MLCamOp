import unittest

import numpy as np

from src.models.view_classifier import ViewClassifier, _preprocess_frame, _FrameDataset


class TestPreprocessFrame(unittest.TestCase):

    def test_output_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = _preprocess_frame(frame)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_output_dtype(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = _preprocess_frame(frame)
        self.assertTrue(tensor.dtype.is_floating_point)


class TestFrameDataset(unittest.TestCase):

    def setUp(self):
        self.frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
        self.labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    def test_length(self):
        ds = _FrameDataset(self.frames, self.labels)
        self.assertEqual(len(ds), 10)

    def test_getitem_returns_tensor_and_label(self):
        ds = _FrameDataset(self.frames, self.labels)
        tensor, label = ds[0]
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(label, 0)

    def test_augmented_dataset_same_shape(self):
        ds = _FrameDataset(self.frames, self.labels, augment=True)
        tensor, label = ds[0]
        self.assertEqual(tensor.shape, (3, 224, 224))


class TestViewClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = ViewClassifier()

    def test_predict_returns_int(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pred = self.classifier.predict(frame)
        self.assertIn(pred, (0, 1))

    def test_predict_batch(self):
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        preds = self.classifier.predict_batch(frames)
        self.assertEqual(len(preds), 5)
        for p in preds:
            self.assertIn(p, (0, 1))


if __name__ == "__main__":
    unittest.main()
