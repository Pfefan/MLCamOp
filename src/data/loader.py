"""Video loading and saving utilities."""

import cv2
import numpy as np


def load_video_data(video_path: str) -> list[np.ndarray]:
    """
    Load all frames from a video file into memory.

    Args:
        video_path: Path to the video file.

    Returns:
        List of frames as numpy arrays (BGR).

    Raises:
        ValueError: If the video file cannot be opened.
    """
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)

    video_capture.release()
    return frames


def load_multiple_videos(video_paths: list[str]) -> dict[str, list[np.ndarray]]:
    """
    Load frames from multiple video files.

    Args:
        video_paths: List of paths to video files.

    Returns:
        Dict mapping path -> list of frames.
    """
    return {path: load_video_data(path) for path in video_paths}


def save_video_data(frames: list[np.ndarray], output_path: str, fps: int = 30) -> None:
    """
    Write a list of frames to a video file.

    Args:
        frames: List of frames as numpy arrays (BGR).
        output_path: Destination file path.
        fps: Frames per second for output video.
    """
    if not frames:
        raise ValueError("Cannot save empty frame list.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()