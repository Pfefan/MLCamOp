"""Video preprocessing utilities: resizing, normalization, frame extraction."""

import cv2
import numpy as np


def preprocess_video(
    input_video_path: str,
    output_video_path: str,
    target_size: tuple[int, int] = (640, 480),
) -> None:
    """
    Resize all frames of a video and write to a new file.

    Args:
        input_video_path: Path to source video.
        output_video_path: Path to write resized video.
        target_size: (width, height) tuple for output frames.
    """
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, target_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, target_size)
        out.write(frame_resized)

    cap.release()
    out.release()


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1].

    Args:
        frame: Input frame as numpy array.

    Returns:
        Normalized frame as float32 array.
    """
    return (frame / 255.0).astype(np.float32)


def extract_frames(video_path: str, frame_interval: int = 30) -> list[np.ndarray]:
    """
    Extract every nth frame from a video, normalized to [0, 1].

    Args:
        video_path: Path to the video file.
        frame_interval: Extract one frame every this many frames.

    Returns:
        List of normalized frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(normalize_frame(frame))
        count += 1

    cap.release()
    return frames