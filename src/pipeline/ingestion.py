"""Pipeline stage: ingest and preprocess raw video files."""

import cv2
import numpy as np


def ingest_video_data(video_paths: list[str]) -> list[np.ndarray | None]:
    """
    Load video data from disk for each provided path.

    Args:
        video_paths: List of file paths to video files.

    Returns:
        List of loaded video objects (currently cv2.VideoCapture handles).
    """
    return [load_video(path) for path in video_paths]


def load_video(path: str) -> cv2.VideoCapture:
    """
    Open a video file and return a VideoCapture object.

    Args:
        path: File path to the video.

    Returns:
        Opened cv2.VideoCapture instance.

    Raises:
        ValueError: If the file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    return cap


def preprocess_video_data(
    video_data: list[cv2.VideoCapture],
) -> list[cv2.VideoCapture]:
    """
    Apply preprocessing to each loaded video.

    Args:
        video_data: List of VideoCapture objects.

    Returns:
        List of preprocessed VideoCapture objects (pass-through until implemented).
    """
    return [preprocess(video) for video in video_data]


def preprocess(video: cv2.VideoCapture) -> cv2.VideoCapture:
    """
    Placeholder preprocessing for a single video.

    Args:
        video: A VideoCapture object.

    Returns:
        The same VideoCapture object (resize/normalize to be added).
    """
    # TODO: Add resizing, normalization, sync offset correction
    return video