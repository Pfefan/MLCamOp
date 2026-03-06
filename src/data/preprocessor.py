"""
Frame-level preprocessing utilities.

These are thin helpers used by the training pipeline.  Heavy video I/O
(opening files, seeking, sampling) lives in ``src.data.sampler``.
"""

import cv2
import numpy as np


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [0.0, 1.0].

    Args:
        frame: Input BGR frame.

    Returns:
        Float32 array with values in [0.0, 1.0].
    """
    return (frame / 255.0).astype(np.float32)


def extract_frames(video_path: str, frame_interval: int = 30) -> list[np.ndarray]:
    """
    Extract every *frame_interval*-th frame from a video, normalized to [0, 1].

    Note: this reads the video sequentially and loads all extracted frames into
    RAM.  For large files prefer ``src.data.sampler.generate_training_data``
    which uses seek-based sampling and disk caching.

    Args:
        video_path:     Path to the video file.
        frame_interval: Keep one frame every this many frames.

    Returns:
        List of normalized float32 frames.

    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(normalize_frame(frame))
        count += 1

    cap.release()
    return frames
