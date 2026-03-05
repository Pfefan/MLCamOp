import unittest
from src.pipeline.ingestion import ingest_videos
from src.pipeline.inference import run_inference
from src.pipeline.editing import edit_video
from src.models.view_classifier import ViewClassifier
from src.models.scene_detector import SceneDetector
from src.models.video_composer import VideoComposer

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.video_paths = ["path/to/video1.mp4", "path/to/video2.mp4"]
        self.view_classifier = ViewClassifier()
        self.scene_detector = SceneDetector()
        self.video_composer = VideoComposer()

    def test_ingest_videos(self):
        videos = ingest_videos(self.video_paths)
        self.assertIsNotNone(videos)
        self.assertEqual(len(videos), len(self.video_paths))

    def test_run_inference(self):
        videos = ingest_videos(self.video_paths)
        predictions = run_inference(videos, self.view_classifier)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(videos))

    def test_edit_video(self):
        videos = ingest_videos(self.video_paths)
        predictions = run_inference(videos, self.view_classifier)
        edited_video = edit_video(videos, predictions, self.scene_detector)
        self.assertIsNotNone(edited_video)

    def test_video_composer(self):
        videos = ingest_videos(self.video_paths)
        predictions = run_inference(videos, self.view_classifier)
        edited_video = edit_video(videos, predictions, self.scene_detector)
        final_video = self.video_composer.compose(edited_video)
        self.assertIsNotNone(final_video)

if __name__ == '__main__':
    unittest.main()