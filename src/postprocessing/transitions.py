"""
Postprocessing: insert visual transitions between cut points.

All functions raise ``NotImplementedError`` until implemented.
The editing stage currently uses hard cuts; these will extend it with
optional fade/dissolve/slide effects between shots.
"""

import numpy as np


def add_fade_transition(
    frames:   list[np.ndarray],
    duration: float,
) -> list[np.ndarray]:
    """
    Add a fade-to-black transition at the end of *frames*.

    Args:
        frames:   Input frame sequence.
        duration: Transition length in seconds.

    Returns:
        Frame sequence with fade appended.
    """
    raise NotImplementedError


def add_slide_transition(
    frames:    list[np.ndarray],
    direction: str,
    duration:  float,
) -> list[np.ndarray]:
    """
    Add a slide transition at the end of *frames*.

    Args:
        frames:    Input frame sequence.
        direction: One of ``'left'``, ``'right'``, ``'up'``, ``'down'``.
        duration:  Transition length in seconds.

    Returns:
        Frame sequence with slide appended.
    """
    raise NotImplementedError


def add_cut_transition(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Return *frames* unchanged (a hard cut requires no inserted frames).

    Args:
        frames: Input frame sequence.

    Returns:
        Same frame sequence.
    """
    return frames


def apply_transitions(
    clips:           list[list[np.ndarray]],
    transition_type: str,
    duration:        float | None = None,
    direction:       str  | None = None,
) -> list[np.ndarray]:
    """
    Join multiple frame sequences with the specified transition between each pair.

    Args:
        clips:           List of frame sequences to join.
        transition_type: ``'fade'``, ``'slide'``, or ``'cut'``.
        duration:        Transition length in seconds (ignored for ``'cut'``).
        direction:       Slide direction (only used for ``'slide'``).

    Returns:
        Single concatenated frame sequence with transitions inserted.
    """
    raise NotImplementedError
