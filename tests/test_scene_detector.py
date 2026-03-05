import unittest
from src.models.scene_detector import SceneDetector

class TestSceneDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SceneDetector()

    def test_scene_detection(self):
        video_path = "path/to/test/video.mp4"
        scenes = self.detector.detect_scenes(video_path)
        self.assertIsInstance(scenes, list)
        self.assertGreater(len(scenes), 0)

    def test_scene_detection_empty_video(self):
        video_path = "path/to/empty/video.mp4"
        scenes = self.detector.detect_scenes(video_path)
        self.assertEqual(scenes, [])

    def test_scene_detection_invalid_video(self):
        video_path = "path/to/invalid/video.mp4"
        with self.assertRaises(ValueError):
            self.detector.detect_scenes(video_path)

if __name__ == '__main__':
    unittest.main()