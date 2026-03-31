"""Assemble the final cut from total-view and close-up frames."""

import numpy as np


def assemble_cut(
    total_view_frames: list[np.ndarray],
    close_up_frames:   list[np.ndarray],
    labels:            list[int],
    min_shot_frames:   int = 10,
) -> list[np.ndarray]:
    """Select wide (label=0) or close-up (label=1) frame at each position."""
    if len(total_view_frames) != len(close_up_frames) != len(labels):
        raise ValueError(
            "total_view_frames, close_up_frames, and labels must all be the same length."
        )

    smoothed = _enforce_min_shot_length(labels, min_shot_frames)

    result: list[np.ndarray] = []
    for i, label in enumerate(smoothed):
        result.append(total_view_frames[i] if label == 0 else close_up_frames[i])

    return result



# ── Private helpers ───────────────────────────────────────────────────────────

def _enforce_min_shot_length(labels: list[int], min_frames: int) -> list[int]:
    """Suppress label runs shorter than *min_frames* by merging with previous."""
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
