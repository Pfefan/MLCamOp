"""
Postprocessing: render the final edited video to disk.

Takes the assembled frame sequence from ``src.pipeline.editing.assemble_cut``
and writes it to an MP4 file, preserving the source video's FPS and resolution.
"""

import cv2
import numpy as np


def render_video(
    frames:      list[np.ndarray],
    output_path: str,
    fps:         float = 25.0,
) -> None:
    """
    Write an assembled frame sequence to an MP4 file.

    Args:
        frames:      Frames to write (all must be the same resolution).
        output_path: Destination file path.
        fps:         Output frames per second.  Should match your source cameras.

    Raises:
        ValueError: If *frames* is empty.
    """
    if not frames:
        raise ValueError("Cannot render empty frame sequence.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
