"""Render an assembled frame sequence to an MP4 file."""

import cv2
import numpy as np


def render_video(
    frames:      list[np.ndarray],
    output_path: str,
    fps:         float = 25.0,
) -> None:
    """Write frames to an MP4 file."""
    if not frames:
        raise ValueError("Cannot render empty frame sequence.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
