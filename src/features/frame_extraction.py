"""Utilities for extracting and saving individual frames from video files."""

import os

import cv2
import numpy as np


def extract_frames(video_path: str, frame_rate: int) -> list[np.ndarray]:
    """
    Extract one frame every `frame_rate` frames from a video.

    Args:
        video_path: Path to the video file.
        frame_rate: Step interval between extracted frames.

    Returns:
        List of frames as numpy arrays (BGR).

    Raises:
        FileNotFoundError: If the video file does not exist.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file at {video_path} does not exist.")

    video_capture = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        if frame_count % frame_rate == 0:
            frames.append(frame)
        frame_count += 1

    video_capture.release()
    return frames


def save_frames(frames: list[np.ndarray], output_dir: str) -> None:
    """
    Save a list of frames as JPEG images to a directory.

    Args:
        frames: List of frames as numpy arrays.
        output_dir: Directory to write frame images into.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)