"""
VideoComposer: assembles the final cut from total-view and close-up frames.

This module is a planned implementation stub.  The composition logic will
use ``ViewClassifier`` predictions to decide which source to pull each frame
from, then ``SceneDetector`` to enforce minimum shot lengths and avoid flicker.

See ``src/pipeline/editing.py`` for the current working implementation.
"""

import numpy as np


class VideoComposer:
    """
    Assemble a final edited video from total-view and close-up frame sequences.

    Not yet implemented — see ``src.pipeline.editing`` for working composition.
    """

    def compose_video(
        self,
        total_view_frames: list[np.ndarray],
        close_up_frames:   list[np.ndarray],
        labels:            list[int],
    ) -> list[np.ndarray]:
        """
        Select frames from total_view or close_up based on classifier labels.

        Args:
            total_view_frames: Wide-shot frames (one per sample point).
            close_up_frames:   Close-up frames at the same timestamps.
            labels:            0 = use wide, 1 = use close-up, per frame.

        Returns:
            Assembled frame sequence.
        """
        raise NotImplementedError

    def add_transitions(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Insert transition frames between detected cut points."""
        raise NotImplementedError

    def render(self, frames: list[np.ndarray], output_path: str, fps: float = 25.0) -> None:
        """Write the assembled frame sequence to a video file."""
        raise NotImplementedError
