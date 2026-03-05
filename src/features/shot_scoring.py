"""Heuristic scoring functions for evaluating shot quality."""

import numpy as np


def score_shots(
    frames: list[np.ndarray], criteria: dict
) -> list[tuple[np.ndarray, float]]:
    """
    Score a list of frames based on provided quality criteria.

    Args:
        frames: List of frames to score.
        criteria: Dict with boolean keys 'focus', 'composition', 'lighting'.

    Returns:
        List of (frame, score) tuples sorted by score descending.
    """
    scored_frames = []

    for frame in frames:
        score = 0.0
        if criteria.get("focus"):
            score += evaluate_focus(frame)
        if criteria.get("composition"):
            score += evaluate_composition(frame)
        if criteria.get("lighting"):
            score += evaluate_lighting(frame)
        scored_frames.append((frame, score))

    scored_frames.sort(key=lambda x: x[1], reverse=True)
    return scored_frames


def evaluate_focus(frame: np.ndarray) -> float:
    """
    Estimate focus quality using variance of pixel differences (approx. Laplacian).

    Higher variance = sharper image.

    Args:
        frame: Input frame as numpy array.

    Returns:
        Focus score as a float.
    """
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    diff = np.diff(gray.astype(np.float32), axis=0)
    return float(np.var(diff))


def evaluate_composition(frame: np.ndarray) -> float:
    """
    Placeholder: score composition (rule of thirds, etc.).

    Args:
        frame: Input frame as numpy array.

    Returns:
        Composition score as a float (currently returns 1.0).
    """
    _ = frame
    return 1.0


def evaluate_lighting(frame: np.ndarray) -> float:
    """
    Score lighting by measuring how close average brightness is to mid-range.

    Args:
        frame: Input frame as numpy array.

    Returns:
        Lighting score — higher is better.
    """
    brightness = float(frame.mean())
    return 1.0 - abs(brightness - 128.0) / 128.0