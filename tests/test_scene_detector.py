import unittest

import numpy as np

from src.models.scene_detector import SceneDetector


class TestSceneDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SceneDetector(threshold=30.0)

    def test_no_scene_change_with_identical_frames(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames = [frame.copy() for _ in range(5)]
        scenes = self.detector.detect_scenes(frames)
        self.assertEqual(scenes, [])

    def test_detects_abrupt_cut(self):
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        frames = [black, black, white, white]
        scenes = self.detector.detect_scenes(frames)
        self.assertIn(2, scenes)

    def test_empty_input(self):
        scenes = self.detector.detect_scenes([])
        self.assertEqual(scenes, [])

    def test_single_frame(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        scenes = self.detector.detect_scenes([frame])
        self.assertEqual(scenes, [])


if __name__ == "__main__":
    unittest.main()
