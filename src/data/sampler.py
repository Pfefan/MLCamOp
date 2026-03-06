"""
Generate labelled training data by comparing total_view and closeup video frames.

Labeling strategy
-----------------
We have three videos recorded simultaneously:
  - total_view.mp4  (wide static camera — always visible)
  - closeup.mp4     (camera operator close-ups — content differs from total)
  - result.mp4      (human-edited final cut — the ground truth)

For each sampled timestamp we read the matching frame from all three videos
and assign the label using a **two-stage** rule:

  Stage 1 — result vs total_view (was the wide shot used?):
    diff(result, total_view) < threshold  →  wide used  →  label = 0

  Stage 2 — result vs closeup (was the closeup used?):
    diff(result, closeup) < threshold     →  closeup used  →  label = 1

  If neither is clearly similar the frame is ambiguous and is **skipped** to
  keep labels clean (transitions, fades, etc.).

This is far more reliable than result-vs-total alone because the old approach
produced ~70 % close-up labels even though only ~30 % of the total_view frames
were close-up content — the result video just happened to differ from the wide
shot at those moments (e.g. a simple edit cut).

Performance
-----------
OpenCV's ``cap.read()`` decodes every frame in sequence — at 25 fps with 1 fps
sampling that means 25 wasted decodes per kept frame across 3 × 7 GB files
(~290 000 full-resolution decodes total ≈ 10–14 minutes).

We avoid this by **seeking directly** to each target frame index with
``cap.set(CAP_PROP_POS_FRAMES, idx)`` and then reading just that one frame.
A single seek + decode of a 1080p frame at thumbnail scale takes ~5 ms;
reading 3 900 frames × 3 videos ≈ 60 s instead of 10–14 min.

Note: random seeking is only reliable on key-frame-aligned positions for some
codecs (H.264 B-frames).  For typical camera footage this is fine; if you see
occasional frame-position drift, encode the video with ``-g 1`` (all I-frames).

Disk caching
------------
Sampled frames and labels are cached to a .pt file keyed by a hash of the
video paths + threshold so the cache is invalidated automatically on any change.
Subsequent runs (evaluate, retrain) load from cache in seconds.
"""

import hashlib
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

logger = logging.getLogger("ConcertVideoEditor")

# Thumbnail size for pixel-diff comparisons (labeling only — never stored)
_THUMB_W, _THUMB_H = 160, 90

# Frames are resized to this before caching and training.
# Matches MobileNetV3 input size: 3900 frames × 224×224×3 ≈ 590 MB vs ~24 GB at 1080p.
_STORE_SIZE = (224, 224)  # (width, height) for cv2.resize


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _thumb(frame: np.ndarray) -> np.ndarray:
    """Resize frame to thumbnail and convert to float32."""
    return cv2.resize(frame, (_THUMB_W, _THUMB_H)).astype(np.float32)


def _pixel_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Mean absolute pixel difference between two BGR frames (thumbnail scale)."""
    return float(np.mean(np.abs(_thumb(frame1) - _thumb(frame2))))


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    return cap


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> tuple[bool, np.ndarray | None]:
    """
    Seek to *frame_idx* and decode exactly that one frame.

    Much faster than sequential ``cap.read()`` when you only need 1-in-N frames:
    avoids decoding all the skipped frames between sample points.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return ret, frame


def _cache_key(total_view_path: str, closeup_path: str, result_path: str,
               sample_fps: float, threshold: float) -> str:
    """Hash of paths + settings so the cache is invalidated on any change."""
    payload = f"{total_view_path}|{closeup_path}|{result_path}|{sample_fps}|{threshold}"
    return hashlib.md5(payload.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_stats(
    total_view_path: str,
    closeup_path:    str,
    result_path:     str,
    sample_fps:      float = 1.0,
    n_samples:       int   = 200,
) -> dict:
    """
    Sample frames and compute pixel-diff statistics to help choose the threshold.

    Computes diffs for both (result vs total_view) and (result vs closeup) so you
    can see the two distributions and pick a threshold between them.

    Uses random seeks instead of sequential reads — runs in seconds, not minutes.

    Args:
        total_view_path: Path to the wide-shot video.
        closeup_path:    Path to the close-up camera video.
        result_path:     Path to the final edited result video.
        sample_fps:      Sampling rate (frames per second) — used to spread samples.
        n_samples:       Number of frames to sample for statistics.

    Returns:
        Dict with keys 'total' and 'closeup', each containing
        'mean', 'std', 'min', 'max', 'p25', 'p50', 'p75'.
    """
    cap_total   = _open_video(total_view_path)
    cap_closeup = _open_video(closeup_path)
    cap_result  = _open_video(result_path)

    video_fps    = cap_total.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps))

    # Sample evenly across the video rather than randomly to avoid clustering
    sample_indices = list(range(0, total_frames, frame_interval))[:n_samples]

    diffs_total:   list[float] = []
    diffs_closeup: list[float] = []

    for idx in sample_indices:
        ret_t, f_total   = _read_frame_at(cap_total,   idx)
        ret_c, f_closeup = _read_frame_at(cap_closeup, idx)
        ret_r, f_result  = _read_frame_at(cap_result,  idx)
        if not (ret_t and ret_c and ret_r):
            continue
        diffs_total.append(_pixel_diff(f_result, f_total))
        diffs_closeup.append(_pixel_diff(f_result, f_closeup))

    cap_total.release()
    cap_closeup.release()
    cap_result.release()

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "p25":  float(np.percentile(arr, 25)),
            "p50":  float(np.percentile(arr, 50)),
            "p75":  float(np.percentile(arr, 75)),
        }

    return {
        "total":   _stats(np.array(diffs_total)),
        "closeup": _stats(np.array(diffs_closeup)),
    }


def generate_training_data(
    total_view_path:      str,
    closeup_path:         str,
    result_path:          str,
    sample_fps:           float = 1.0,
    similarity_threshold: float = 65.9,
    cache_dir:            str   = "data/cache",
    force_rebuild:        bool  = False,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Generate (frame, label) pairs using total_view, closeup, and result videos.

    Frames are sourced from total_view only (the wide shot is the model input).
    Labels are derived by comparing result to both cameras:
      - result ≈ total_view  →  label 0 (wide shot)
      - result ≈ closeup     →  label 1 (close-up)
      - ambiguous            →  skipped (keeps label quality high)

    Results are cached to disk; re-reading three 7 GB videos takes ~14 minutes.
    Subsequent calls with the same settings load from cache in seconds.

    Args:
        total_view_path:      Path to the wide-shot video.
        closeup_path:         Path to the close-up camera video.
        result_path:          Path to the final edited result video.
        sample_fps:           Frames per second to sample (1.0 → ~3900 per hour).
        similarity_threshold: Pixel diff below this = "same shot".  Run
                              ``scripts/tune_threshold.py`` to find the right value.
        cache_dir:            Directory where cached tensors are stored.
        force_rebuild:        If True, ignore any existing cache.

    Returns:
        Tuple of (frames, labels):
            frames: list of BGR np.ndarray frames taken from total_view.
            labels: list of int — 0 = wide shot, 1 = close-up.

    Raises:
        ValueError: If any video file cannot be opened.
    """
    # ── Cache lookup ──────────────────────────────────────────────────────────
    os.makedirs(cache_dir, exist_ok=True)
    key        = _cache_key(total_view_path, closeup_path, result_path,
                            sample_fps, similarity_threshold)
    cache_path = Path(cache_dir) / f"frames_{key}.pt"

    if not force_rebuild and cache_path.exists():
        logger.info("Loading cached training data from %s", cache_path)
        data = torch.load(str(cache_path), weights_only=False)
        frames = [f.numpy() for f in data["frames"]]
        labels = data["labels"].tolist()
        _print_label_summary(labels)
        return frames, labels

    # ── Video reading ─────────────────────────────────────────────────────────
    cap_total   = _open_video(total_view_path)
    cap_closeup = _open_video(closeup_path)
    cap_result  = _open_video(result_path)

    video_fps      = cap_total.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames   = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps))

    # Pre-compute all target frame indices so we can seek directly to each one.
    # This avoids decoding the N-1 frames between sample points — the single
    # biggest performance win (25× faster for 1 fps sampling of 25 fps video).
    sample_indices = list(range(0, total_frames, frame_interval))
    n_samples_expected = len(sample_indices)

    logger.info(
        "Video FPS: %.1f | Seeking to every %d-th frame | %d sample points",
        video_fps, frame_interval, n_samples_expected,
    )
    print(f"Video FPS: {video_fps:.1f} | Sampling every {frame_interval} frames")
    print(f"Total frames in video: {total_frames} → ~{n_samples_expected} samples")

    frames:  list[np.ndarray] = []
    labels:  list[int]        = []
    skipped: int              = 0

    iterable = (
        tqdm(sample_indices, desc="Sampling frames", unit="frame")
        if _TQDM_AVAILABLE
        else sample_indices
    )

    for idx in iterable:
        ret_t, f_total   = _read_frame_at(cap_total,   idx)
        ret_c, f_closeup = _read_frame_at(cap_closeup, idx)
        ret_r, f_result  = _read_frame_at(cap_result,  idx)

        if not (ret_t and ret_c and ret_r):
            continue

        diff_to_total   = _pixel_diff(f_result, f_total)
        diff_to_closeup = _pixel_diff(f_result, f_closeup)

        wide_match    = diff_to_total   < similarity_threshold
        closeup_match = diff_to_closeup < similarity_threshold

        if wide_match and not closeup_match:
            # Result clearly matches total_view → wide shot.
            # Resize to model input size before storing — 224×224 × 3 bytes per
            # frame vs 1080p × 3 = ~590 MB total instead of ~24 GB for 3900 frames.
            frames.append(cv2.resize(f_total, _STORE_SIZE))
            labels.append(0)
        elif closeup_match and not wide_match:
            # Result clearly matches closeup → close-up shot
            frames.append(cv2.resize(f_total, _STORE_SIZE))
            labels.append(1)
        else:
            # Ambiguous (transition, fade, or matching both) — skip
            skipped += 1

    cap_total.release()
    cap_closeup.release()
    cap_result.release()

    if not frames:
        raise RuntimeError(
            "No labelled frames generated.  Check video paths and run "
            "scripts/tune_threshold.py to calibrate similarity_threshold."
        )

    if skipped > 0:
        logger.info("Skipped %d ambiguous frames (transitions/fades)", skipped)

    # ── Cache to disk ─────────────────────────────────────────────────────────
    torch.save(
        {
            "frames": [torch.from_numpy(f) for f in frames],
            "labels": torch.tensor(labels, dtype=torch.long),
        },
        str(cache_path),
    )
    logger.info("Cached %d labelled frames to %s", len(frames), cache_path)

    _print_label_summary(labels)
    return frames, labels


def _print_label_summary(labels: list[int]) -> None:
    total   = len(labels)
    n_wide  = labels.count(0)
    n_close = labels.count(1)
    print(
        f"Sampled {total} frames: "
        f"{n_wide} wide ({n_wide / total * 100:.1f}%), "
        f"{n_close} close-up ({n_close / total * 100:.1f}%)"
    )
