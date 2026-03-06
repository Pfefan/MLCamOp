"""
Low-level video I/O helpers.

Prefer ``src.data.loader`` for all new code — it has better error handling and
full type annotations.  The functions here exist for backward compatibility with
any code that imported from this module before the refactor.
"""

import cv2
import numpy as np


def read_video(file_path: str) -> list[np.ndarray]:
    """
    Load all frames from a video file into memory.

    Warning: loads the *entire* video into RAM.  For long/large files use
    ``src.data.sampler.generate_training_data`` which streams with seeking.

    Args:
        file_path: Path to the video file.

    Returns:
        List of BGR frames as numpy arrays.

    Raises:
        ValueError: If the file cannot be opened.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {file_path}")

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def write_video(frames: list[np.ndarray], output_path: str, fps: float = 30.0) -> None:
    """
    Write a list of BGR frames to a video file.

    Args:
        frames: Frames to write (must all be the same size).
        output_path: Destination file path.
        fps: Output frames per second.

    Raises:
        ValueError: If *frames* is empty.
    """
    if not frames:
        raise ValueError("Cannot write empty frame list.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
