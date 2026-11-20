"""Streaming inference over a video or dual-camera source."""

from collections import deque

import cv2
import numpy as np

from src.data.audio import extract_frame_audio_features
from src.models.view_classifier import ViewClassifier, _causal_window_indices
from src.utils.metrics import viterbi_smooth


def _pad_window(frames: list[np.ndarray], window: int) -> list[np.ndarray]:
    """Left-pad a short buffer to the expected window size."""
    if len(frames) < window:
        frames = [frames[0]] * (window - len(frames)) + frames
    return frames


def classify_video(
    total_view_path: str,
    classifier:      ViewClassifier,
    sample_fps:      float = 1.0,
    closeup_path:    str | None = None,
    return_proba:    bool = False,
) -> list[tuple[float, int]] | tuple[list[float], np.ndarray]:
    """Classify sampled frames.

    Returns (timestamp, label) pairs by default, or (timestamps, probabilities)
    when *return_proba* is set (used for sequence smoothing).
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
    probs:      list[np.ndarray] = []

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

            if sampled >= lookahead:
                t = sampled - lookahead
                probs.append(classifier.predict_window_proba(
                    _pad_window(list(frame_buf), window), audio_window(t),
                ))
            sampled += 1
        idx += 1

    cap_total.release()
    if cap_closeup is not None:
        cap_closeup.release()

    if not timestamps:
        return ([], np.zeros((0, classifier.num_classes), dtype=np.float32)) if return_proba else []

    buf_list = list(frame_buf)
    last     = buf_list[-1]
    for d in range(1, lookahead + 1):
        t = (sampled - 1) - lookahead + d
        probs.append(classifier.predict_window_proba(
            _pad_window(buf_list[d:] + [last] * d, window), audio_window(t),
        ))

    prob_arr = np.asarray(probs, dtype=np.float32)
    if return_proba:
        return timestamps, prob_arr
    return list(zip(timestamps, prob_arr.argmax(axis=1).astype(int).tolist()))


def classify_video_smoothed(
    total_view_path: str,
    classifier:      ViewClassifier,
    sample_fps:      float = 1.0,
    closeup_path:    str | None = None,
    cut_penalty:     float = 0.5,
    min_shot:        int   = 1,
) -> list[tuple[float, int]]:
    """Classify then Viterbi-smooth the label sequence, matching the eval path."""
    timestamps, probs = classify_video(
        total_view_path, classifier, sample_fps=sample_fps,
        closeup_path=closeup_path, return_proba=True,
    )
    if not timestamps:
        return []
    labels = viterbi_smooth(probs, cut_penalty, min_shot=min_shot)
    return list(zip(timestamps, labels))
