"""Generate labelled training data by comparing total_view and closeup video frames.

Labeling: for each sampled timestamp, compare the human-edited result video against
both the wide and closeup cameras. Whichever matches (pixel diff < threshold) becomes
the label. Ambiguous frames (both or neither match) are skipped.

Frames are cached to disk (keyed by paths + settings) so reruns load in seconds.
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
    """Seek to *frame_idx* and decode that one frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    return cap.read()


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
    """Sample frames and compute pixel-diff statistics for threshold tuning."""
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
    """Generate (frame, label) pairs from three synchronised concert videos.

    Frames are from total_view (224x224). Labels from comparing result to both cameras.
    Results are cached to disk keyed by paths + settings.
    """
    # ── Cache lookup ────────────────────────────────────────────────────────
    os.makedirs(cache_dir, exist_ok=True)
    key        = _cache_key(total_view_path, closeup_path, result_path,
                            sample_fps, similarity_threshold)
    cache_path = Path(cache_dir) / f"frames_{key}.pt"

    if not force_rebuild and cache_path.exists():
        logger.info("Loading cached training data from %s", cache_path)
        data = torch.load(str(cache_path), weights_only=False)

        # Support both old (list-of-tensors) and new (stacked tensor) cache formats
        raw = data["frames"]
        if isinstance(raw, list):
            frames = [f.numpy() for f in raw]
        else:
            frames = list(raw.numpy())

        labels = data["labels"].tolist()
        del data, raw
        _print_label_summary(labels)
        return frames, labels

    # ── Video reading ────────────────────────────────────────────────────────
    cap_total   = _open_video(total_view_path)
    cap_closeup = _open_video(closeup_path)
    cap_result  = _open_video(result_path)

    video_fps      = cap_total.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames   = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps))
    sample_indices = list(range(0, total_frames, frame_interval))

    logger.info(
        "Video FPS: %.1f | Every %d-th frame | %d sample points",
        video_fps, frame_interval, len(sample_indices),
    )

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
            frames.append(cv2.resize(f_total, _STORE_SIZE))
            labels.append(0)
        elif closeup_match and not wide_match:
            frames.append(cv2.resize(f_total, _STORE_SIZE))
            labels.append(1)
        else:
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

    # Stack into a single tensor for efficient serialisation (one memcpy vs 44k pickle calls)
    torch.save(
        {
            "frames": torch.from_numpy(np.stack(frames)),
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


def split_dataset(
    frames:    list[np.ndarray],
    labels:    list[int],
    val_split: float = 0.2,
    seed:      int   = 42,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
    """Shuffle and split frames/labels into train/val sets."""
    n       = len(frames)
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n)

    split        = int(n * (1 - val_split))
    train_idx    = indices[:split]
    val_idx      = indices[split:]

    train_frames = [frames[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_frames   = [frames[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    return train_frames, train_labels, val_frames, val_labels


def concert_split(
    concerts_frames: list[list[np.ndarray]],
    concerts_labels: list[list[int]],
    val_concert_idx: int = -1,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
    """Split by concert: train on all except one, validate on the held-out one."""
    n = len(concerts_frames)
    if n < 2:
        raise ValueError(
            "concert_split() requires at least 2 concerts. "
            "Use split_dataset() for single-concert training."
        )

    idx = val_concert_idx % n   # support negative indexing

    train_frames: list[np.ndarray] = []
    train_labels: list[int]        = []
    for i in range(n):
        if i != idx:
            train_frames.extend(concerts_frames[i])
            train_labels.extend(concerts_labels[i])

    val_frames = concerts_frames[idx]
    val_labels = concerts_labels[idx]

    return train_frames, train_labels, val_frames, val_labels
