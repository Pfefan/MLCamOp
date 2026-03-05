import unittest
from src.data.loader import load_video_data

class TestLoader(unittest.TestCase):

    def test_load_video_data_valid(self):
        # Test loading valid video data
        video_path = 'path/to/valid/video.mp4'
        data = load_video_data(video_path)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_load_video_data_invalid(self):
        # Test loading invalid video data
        video_path = 'path/to/invalid/video.mp4'
        with self.assertRaises(FileNotFoundError):
            load_video_data(video_path)

    def test_load_video_data_empty(self):
        # Test loading from an empty path
        video_path = ''
        with self.assertRaises(ValueError):
            load_video_data(video_path)

if __name__ == '__main__':
    unittest.main()