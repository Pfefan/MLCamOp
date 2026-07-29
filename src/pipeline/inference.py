"""Streaming ViewClassifier inference over a video (or dual-camera) source.

Frames are decoded sequentially and held only in a bounded ring buffer sized to
the classifier's causal temporal window, so memory does not grow with video
length and no code path depends on more than `classifier.lookahead` future
frames — keeping this compatible with live / real-time use.
"""

from collections import deque

import cv2
import numpy as np

from src.data.audio import extract_frame_audio_features
from src.models.view_classifier import ViewClassifier, _causal_window_indices


def _pad_window(frames: list[np.ndarray], window: int) -> list[np.ndarray]:
    """Left-pad a short buffer to `window` frames by repeating the oldest frame."""
    if len(frames) < window:
        frames = [frames[0]] * (window - len(frames)) + frames
    return frames


def classify_video(
    total_view_path: str,
    classifier:      ViewClassifier,
    sample_fps:      float = 1.0,
    closeup_path:    str | None = None,
) -> list[tuple[float, int]]:
    """Classify sampled frames, returning (timestamp, label) pairs in order.

    If closeup_path is given, both cameras are decoded in lockstep and stacked as
    6-channel dual frames. Decoding is sequential (grab/retrieve) and streaming:
    only the last `window` sampled frames are kept in memory. When the model has
    an audio branch, per-sample log-mel features are precomputed (tiny) and windowed
    causally alongside the frames.
    """
    cap_total = cv2.VideoCapture(total_view_path)
    if not cap_total.isOpened():
        raise ValueError(f"Cannot open video: {total_view_path}")

    cap_closeup = None
    if closeup_path:
        cap_closeup = cv2.VideoCapture(closeup_path)
        if not cap_closeup.isOpened():
            raise ValueError(f"Cannot open video: {closeup_path}")

    video_fps = cap_total.get(cv2.CAP_PROP_FPS) or 25.0
    step      = max(1, int(video_fps / sample_fps))
    window    = classifier.window
    lookahead = classifier.lookahead

    audio_feats = None
    if classifier.audio_dim > 0:
        total_frames = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_times = [i / video_fps for i in range(0, total_frames, step)]
        audio_feats  = extract_frame_audio_features(
            total_view_path, sample_times, classifier.audio_sample_rate, classifier.audio_dim,
        )

    def audio_window(t: int) -> np.ndarray | None:
        if audio_feats is None:
            return None
        indices = _causal_window_indices(t, len(audio_feats), window, lookahead)
        return audio_feats[indices]

    frame_buf: deque[np.ndarray] = deque(maxlen=window)
    timestamps: list[float]      = []
    preds:      list[int]        = []

    idx     = 0
    sampled = 0
    while True:
        if not cap_total.grab():
            break
        if cap_closeup is not None and not cap_closeup.grab():
            break

        if idx % step == 0:
            ok_t, f_total = cap_total.retrieve()
            if not ok_t:
                break
            if cap_closeup is not None:
                ok_c, f_closeup = cap_closeup.retrieve()
                if not ok_c:
                    break
                frame = np.concatenate(
                    [cv2.resize(f_total, (224, 224)), cv2.resize(f_closeup, (224, 224))],
                    axis=2,
                )
            else:
                frame = f_total

            frame_buf.append(frame)
            timestamps.append(idx / video_fps)

            # Decision for sampled index `sampled - lookahead` is ready once its
            # bounded lookahead of future frames has been buffered.
            if sampled >= lookahead:
                t = sampled - lookahead
                preds.append(classifier.predict_window(
                    _pad_window(list(frame_buf), window), audio_window(t),
                ))
            sampled += 1
        idx += 1

    cap_total.release()
    if cap_closeup is not None:
        cap_closeup.release()

    if not timestamps:
        return []

    # Flush the final `lookahead` decisions, clamping the missing future frames
    # to the last available frame (matching training's edge-clamped window).
    buf_list = list(frame_buf)
    last     = buf_list[-1]
    for d in range(1, lookahead + 1):
        t = (sampled - 1) - lookahead + d
        preds.append(classifier.predict_window(
            _pad_window(buf_list[d:] + [last] * d, window), audio_window(t),
        ))

    return list(zip(timestamps, preds))
