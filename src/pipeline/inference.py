"""Run ViewClassifier inference over a video stream."""

import cv2
import numpy as np

from src.models.view_classifier import ViewClassifier


def classify_video(
    total_view_path: str,
    classifier:      ViewClassifier,
    sample_fps:      float = 1.0,
) -> list[tuple[float, int]]:
    """Run the classifier on sampled frames, return (timestamp, label) pairs."""
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
