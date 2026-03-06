"""
Pipeline stage: assemble the final cut from total-view and close-up frames.

The editing stage takes per-frame classifier labels and produces a single
output frame sequence: wide frames where label=0, close-up frames where
label=1.  A minimum shot length is enforced to prevent rapid flicker.
"""

import cv2
import numpy as np


def assemble_cut(
    total_view_frames: list[np.ndarray],
    close_up_frames:   list[np.ndarray],
    labels:            list[int],
    min_shot_frames:   int = 10,
) -> list[np.ndarray]:
    """
    Build the output frame sequence by selecting the correct source per label.

    Labels of 0 pull from *total_view_frames*; labels of 1 pull from
    *close_up_frames*.  A minimum shot length is enforced so the edit doesn't
    flicker when the classifier oscillates on a single frame.

    Args:
        total_view_frames: Wide-shot frames.  Must be same length as *labels*.
        close_up_frames:   Close-up frames at the same timestamps.
        labels:            Per-frame predictions — 0 = wide, 1 = close-up.
        min_shot_frames:   Minimum number of consecutive frames before a cut
                           is allowed (suppresses single-frame flicker).

    Returns:
        Assembled frame sequence ready to be written to disk.
    """
    if len(total_view_frames) != len(close_up_frames) != len(labels):
        raise ValueError(
            "total_view_frames, close_up_frames, and labels must all be the same length."
        )

    smoothed = _enforce_min_shot_length(labels, min_shot_frames)

    result: list[np.ndarray] = []
    for i, label in enumerate(smoothed):
        result.append(total_view_frames[i] if label == 0 else close_up_frames[i])

    return result


def save_video(
    frames:      list[np.ndarray],
    output_path: str,
    fps:         float = 25.0,
) -> None:
    """
    Write an assembled frame sequence to a video file.

    Args:
        frames:      Frames to write (must all be the same resolution).
        output_path: Destination file path (MP4).
        fps:         Output frames per second.  Should match source FPS.

    Raises:
        ValueError: If *frames* is empty.
    """
    if not frames:
        raise ValueError("Cannot save empty frame list.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()


# ── Private helpers ───────────────────────────────────────────────────────────

def _enforce_min_shot_length(labels: list[int], min_frames: int) -> list[int]:
    """
    Suppress label flips that last fewer than *min_frames* frames.

    Walks through the label sequence and converts isolated short runs back to
    the surrounding majority label, preventing rapid-fire cuts in output.

    Args:
        labels:     Per-frame label list.
        min_frames: Minimum run length to keep.

    Returns:
        Smoothed label list of the same length.
    """
    if not labels or min_frames <= 1:
        return list(labels)

    smoothed = list(labels)
    i = 0
    while i < len(smoothed):
        current = smoothed[i]
        run_end = i + 1
        while run_end < len(smoothed) and smoothed[run_end] == current:
            run_end += 1
        run_length = run_end - i
        if run_length < min_frames and i > 0:
            # Replace this short run with the previous label
            prev_label = smoothed[i - 1]
            for j in range(i, run_end):
                smoothed[j] = prev_label
        i = run_end

    return smoothed
