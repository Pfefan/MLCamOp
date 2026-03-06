"""
Audio analysis helpers for concert video editing.

These functions are planned for a future phase where audio cues (beat onsets,
applause detection, silence) inform cut decisions.  All functions raise
``NotImplementedError`` until implemented.

Planned dependency: ``librosa`` for beat tracking and spectral analysis.
Install with: ``pip install librosa soundfile``
"""


def analyze_audio(audio_path: str) -> dict:
    """
    Extract high-level audio features from a concert recording.

    Planned features: tempo, beat positions, onset strength, loudness envelope.

    Args:
        audio_path: Path to audio file (WAV or MP3) or video file with audio track.

    Returns:
        Dict of feature arrays keyed by feature name.
    """
    raise NotImplementedError


def extract_audio_features(audio_data: dict) -> dict:
    """
    Derive editing-relevant signals from raw audio features.

    Converts raw spectral/temporal features into signals that can drive cut
    decisions: beat-aligned timestamps, applause segments, silence gaps.

    Args:
        audio_data: Output of ``analyze_audio()``.

    Returns:
        Dict with keys such as ``'beat_times'``, ``'applause_mask'``,
        ``'silence_mask'``.
    """
    raise NotImplementedError


def sync_audio_with_video(
    audio_features: dict,
    total_frames:   int,
    fps:            float,
) -> list[float]:
    """
    Map audio-derived timestamps to per-frame scores for cut-point weighting.

    Returns a float score per frame: higher = better moment to cut.

    Args:
        audio_features: Output of ``extract_audio_features()``.
        total_frames:   Number of frames in the video.
        fps:            Video frames per second.

    Returns:
        List of float scores, one per frame.
    """
    raise NotImplementedError
