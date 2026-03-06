"""
Heuristic scene-change detector based on frame-to-frame pixel differences.

This is used as a lightweight fallback / sanity-check layer.  The primary
shot classification is done by ``ViewClassifier``; ``SceneDetector`` flags
hard cuts so the composer can avoid splitting a run of identical shots.
"""

import cv2
import numpy as np


class SceneDetector:
    """
    Detect scene changes by comparing consecutive frame histograms.

    A scene change is declared when the mean absolute difference between
    the grayscale thumbnails of two consecutive frames exceeds *threshold*.

    Args:
        threshold: Pixel-diff score above which a cut is declared (0–255 scale).
                   Lower = more sensitive.  Tune with ``tune_threshold.py``.
    """

    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def detect_scenes(self, frames: list[np.ndarray]) -> list[int]:
        """
        Return the frame indices where a scene change was detected.

        Args:
            frames: Sequence of BGR frames in display order.

        Returns:
            List of indices *i* where a cut occurs between frames[i-1] and frames[i].
        """
        scene_changes: list[int] = []

        for i in range(1, len(frames)):
            if self._is_scene_change(frames[i - 1], frames[i]):
                scene_changes.append(i)

        return scene_changes

    # ── Private ───────────────────────────────────────────────────────────────

    def _is_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Return True if the pixel difference between two frames exceeds threshold."""
        return self._frame_diff(frame1, frame2) > self.threshold

    @staticmethod
    def _frame_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Mean absolute difference between two grayscale thumbnail frames.

        Downsampling to 160×90 before comparison is fast and sufficient for
        detecting hard cuts.
        """
        size = (160, 90)
        g1 = cv2.cvtColor(cv2.resize(frame1, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(cv2.resize(frame2, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.mean(np.abs(g1 - g2)))
