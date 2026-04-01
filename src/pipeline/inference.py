"""Run ViewClassifier inference over a video stream."""

import cv2
import numpy as np

from src.models.view_classifier import ViewClassifier


def classify_video(
    total_view_path: str,
    classifier:      ViewClassifier,
    sample_fps:      float = 1.0,
    closeup_path:    str | None = None,
) -> list[tuple[float, int]]:
    """Run the classifier on sampled frames, return (timestamp, label) pairs.

    If closeup_path is provided, feeds both cameras side-by-side (dual-frame mode).
    """
    cap_total = cv2.VideoCapture(total_view_path)
    if not cap_total.isOpened():
        raise ValueError(f"Cannot open video: {total_view_path}")

    cap_closeup = None
    if closeup_path:
        cap_closeup = cv2.VideoCapture(closeup_path)
        if not cap_closeup.isOpened():
            raise ValueError(f"Cannot open video: {closeup_path}")

    video_fps    = cap_total.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))
    step         = max(1, int(video_fps / sample_fps))

    results: list[tuple[float, int]] = []

    for idx in range(0, total_frames, step):
        cap_total.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_t, f_total = cap_total.read()
        if not ret_t:
            break

        if cap_closeup is not None:
            cap_closeup.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_c, f_closeup = cap_closeup.read()
            if not ret_c:
                break
            # 6-channel: each camera resized to 224×224, stacked on channel axis
            wide_resized  = cv2.resize(f_total,   (224, 224))
            close_resized = cv2.resize(f_closeup, (224, 224))
            frame = np.concatenate([wide_resized, close_resized], axis=2)
        else:
            frame = f_total

        label     = classifier.predict(frame)
        timestamp = idx / video_fps
        results.append((timestamp, label))

    cap_total.release()
    if cap_closeup is not None:
        cap_closeup.release()
    return results
