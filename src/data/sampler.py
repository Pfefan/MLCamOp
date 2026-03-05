"""
Generate labelled training data by comparing total_view and result video frames.

How it works:
    For each sampled timestamp, we read the same frame from both:
      - total_view.mp4  (your wide camera, always recording everything)
      - result.mp4      (the final human-edited video)

    If those two frames look the same  → the editor chose the wide shot  → label 0
    If those two frames look different → the editor cut to close-up       → label 1

This lets us use the human-edited video as our "ground truth" without any manual labelling.
"""

import cv2
import numpy as np


def _frames_are_similar(
    frame1: np.ndarray,
    frame2: np.ndarray,
    threshold: float = 30.0,
) -> bool:
    """
    Compare two frames using mean absolute pixel difference on small thumbnails.

    We resize to 160x90 first for speed — we only need a rough similarity check.

    Args:
        frame1: First frame (BGR).
        frame2: Second frame (BGR).
        threshold: Mean pixel diff below this = frames are considered the same.

    Returns:
        True if frames are visually similar (wide shot used), False otherwise.
    """
    f1 = cv2.resize(frame1, (160, 90)).astype(np.float32)
    f2 = cv2.resize(frame2, (160, 90)).astype(np.float32)
    return float(np.mean(np.abs(f1 - f2))) < threshold


def generate_training_data(
    total_view_path: str,
    result_path: str,
    sample_fps: float = 1.0,
    similarity_threshold: float = 30.0,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Generate (frame, label) pairs from the total_view and result videos.

    Reads both videos in sync and samples at `sample_fps` frames per second.
    At 1 fps on a 65-minute concert: ~3900 samples — fits easily in RAM.

    Args:
        total_view_path: Path to the wide-shot video file.
        result_path: Path to the final edited/result video file.
        sample_fps: How many frames per second to sample (1.0 is fine to start).
        similarity_threshold: Pixel diff threshold for wide vs close-up decision.

    Returns:
        Tuple of (frames, labels):
            frames: List of BGR np.ndarray frames from total_view.
            labels: List of ints — 0 = wide shot, 1 = close-up.

    Raises:
        ValueError: If either video file cannot be opened.
    """
    cap_total = cv2.VideoCapture(total_view_path)
    cap_result = cv2.VideoCapture(result_path)

    if not cap_total.isOpened():
        raise ValueError(f"Cannot open total_view video: {total_view_path}")
    if not cap_result.isOpened():
        raise ValueError(f"Cannot open result video: {result_path}")

    video_fps = cap_total.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))
    # How many native frames to skip between each sample
    frame_interval = max(1, int(video_fps / sample_fps))

    frames: list[np.ndarray] = []
    labels: list[int] = []
    frame_idx = 0

    print(f"Video FPS: {video_fps:.1f} | Sampling every {frame_interval} frames")
    print(f"Total frames in video: {total_frames} → ~{total_frames // frame_interval} samples")

    while True:
        ret_t, frame_total = cap_total.read()
        ret_r, frame_result = cap_result.read()

        if not ret_t or not ret_r:
            break

        if frame_idx % frame_interval == 0:
            is_wide = _frames_are_similar(frame_total, frame_result, similarity_threshold)
            label = 0 if is_wide else 1
            frames.append(frame_total)
            labels.append(label)

        frame_idx += 1

    cap_total.release()
    cap_result.release()

    n_wide = labels.count(0)
    n_close = labels.count(1)
    print(
        f"Sampled {len(frames)} frames: "
        f"{n_wide} wide ({n_wide/len(frames)*100:.1f}%), "
        f"{n_close} close-up ({n_close/len(frames)*100:.1f}%)"
    )

    return frames, labels
