"""Render the final edited video from assembled frame sequences."""

import cv2
import numpy as np


def render_final_video(
    total_view_frames: list[np.ndarray],
    close_up_frames: list[np.ndarray],
    output_path: str,
    fps: float = 30.0,
) -> None:
    """
    Combine total-view and close-up frames and write to a video file.

    Args:
        total_view_frames: Frames from the wide camera.
        close_up_frames: Frames from the close-up camera.
        output_path: Destination file path.
        fps: Output frames per second.
    """
    final_frames: list[np.ndarray] = []

    for total_frame, close_frame in zip(total_view_frames, close_up_frames):
        final_frames.append(total_frame)
        final_frames.append(close_frame)

    if not final_frames:
        return

    height, width = final_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in final_frames:
        out.write(frame)

    out.release()