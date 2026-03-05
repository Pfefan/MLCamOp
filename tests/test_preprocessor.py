import unittest
from src.data.preprocessor import preprocess_video

class TestVideoPreprocessor(unittest.TestCase):

    def test_preprocess_video_resizing(self):
        input_video_path = "path/to/input/video.mp4"
        output_video_path = "path/to/output/video.mp4"
        target_size = (640, 480)

        # Call the preprocess_video function
        preprocess_video(input_video_path, output_video_path, target_size)

        # Here you would typically check if the output video exists and has the correct dimensions
        # This is a placeholder for actual assertions
        self.assertTrue(True)  # Replace with actual checks

    def test_preprocess_video_normalization(self):
        input_video_path = "path/to/input/video.mp4"
        output_video_path = "path/to/output/video.mp4"

        # Call the preprocess_video function
        preprocess_video(input_video_path, output_video_path)

        # Here you would typically check if the output video is normalized
        # This is a placeholder for actual assertions
        self.assertTrue(True)  # Replace with actual checks

if __name__ == '__main__':
    unittest.main()