"""
Pipeline stage: run ViewClassifier inference over a video stream.

Feeds frames from ``total_view`` through the trained model and returns a
per-frame label sequence (0 = wide, 1 = close-up) that the editing stage
uses to decide which source to pull from at each moment.
"""

import cv2
import numpy as np

from src.models.view_classifier import ViewClassifier


def classify_video(
    total_view_path: str,
    classifier:      ViewClassifier,
    sample_fps:      float = 1.0,
) -> list[tuple[float, int]]:
    """
    Run the classifier over a video and return (timestamp_sec, label) pairs.

    Seeks directly to each sample point rather than reading every frame so it
    stays fast even on 7 GB files.

    Args:
        total_view_path: Path to the wide-shot video.
        classifier:      Trained ``ViewClassifier`` instance.
        sample_fps:      How many frames per second to classify.

    Returns:
        List of ``(timestamp_seconds, label)`` — label is 0 (wide) or 1 (close-up).

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(total_view_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {total_view_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step         = max(1, int(video_fps / sample_fps))

    results: list[tuple[float, int]] = []

    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        label     = classifier.predict(frame)
        timestamp = idx / video_fps
        results.append((timestamp, label))

    cap.release()
    return results
