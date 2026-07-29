"""Generate labelled training data by comparing total_view and closeup video frames.

Labeling: for each sampled timestamp, compare the human-edited result video against
both the wide and closeup cameras. Whichever matches (pixel diff < threshold) becomes
the label. Ambiguous frames (both or neither match) are skipped.

Frames are cached to disk (keyed by paths + settings) so reruns load in seconds.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.audio import extract_frame_audio_features

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


@dataclass
class SampleSet:
    """Per-sample training arrays that must stay index-aligned through splitting.

    frames:  (N, H, W, C) uint8 video frames.
    labels:  hard nearest-camera class per frame (0 = wide, 1 = close-up).
    weights: per-sample loss weight; transition frames get < 1 (soft/uncertain).
    audio:   (N, n_mels) log-mel features, or None when the audio branch is off.
    """

    frames:  np.ndarray
    labels:  list[int]
    weights: list[float]
    audio:   np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.frames)


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
    dual_frame:           bool  = True,
    keep_transitions:     bool  = True,
    transition_weight:    float = 0.3,
    audio_enabled:        bool  = False,
    audio_sample_rate:    int   = 16000,
    audio_n_mels:         int   = 32,
) -> SampleSet:
    """Build a labelled SampleSet from three synchronised concert videos.

    Frames are wide+closeup stacked as 6-channel (224×224×6) when dual_frame=True,
    else total_view only. Each frame is labelled by whichever camera the edited
    result matches. Ambiguous transition frames (fades/dissolves) are kept when
    keep_transitions=True with a lower `transition_weight`, so the model sees real
    cut moments instead of only clean steady state. Results are cached to disk.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key_parts = ["dual" if dual_frame else "single"]
    if keep_transitions:
        key_parts.append(f"tr{transition_weight}")
    if audio_enabled:
        key_parts.append(f"aud{audio_sample_rate}x{audio_n_mels}")
    key = _cache_key(total_view_path, closeup_path, result_path,
                     sample_fps, similarity_threshold) + "_" + "_".join(key_parts)
    cache_path = Path(cache_dir) / f"frames_{key}.pt"

    if not force_rebuild and cache_path.exists():
        logger.info("Loading cached training data from %s", cache_path)
        # mmap=True: only pages accessed are loaded from disk — the full
        # cache file is NOT read into RAM upfront.  Saves ~19 GB of initial
        # allocation when loading; the OS will page in frames on demand.
        try:
            data = torch.load(str(cache_path), weights_only=False, mmap=True)
        except TypeError:
            # mmap kwarg requires PyTorch >= 2.0; fall back gracefully
            data = torch.load(str(cache_path), weights_only=False)

        # Support both old (list-of-tensors) and new (stacked tensor) formats.
        # Always return a single contiguous ndarray (N, H, W, C) — NOT a
        # list — so _FrameDataset is COW-safe across DataLoader workers.
        raw = data["frames"]
        if isinstance(raw, list):
            cached_frames = np.stack([f.numpy() for f in raw])
        else:
            cached_frames = np.ascontiguousarray(raw.numpy())

        labels  = data["labels"].tolist()
        weights = data["weights"].tolist() if "weights" in data else [1.0] * len(labels)
        audio   = data["audio"].numpy() if data.get("audio") is not None else None
        del data, raw
        _print_label_summary(labels)
        return SampleSet(cached_frames, labels, weights, audio)

    # ── Video reading ────────────────────────────────────────────────────────
    cap_total   = _open_video(total_view_path)
    cap_closeup = _open_video(closeup_path)
    cap_result  = _open_video(result_path)

    video_fps      = cap_total.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames   = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps))
    sample_indices = list(range(0, total_frames, frame_interval))

    logger.info(
        "Video FPS: %.1f | Every %d-th frame | %d sample points%s",
        video_fps, frame_interval, len(sample_indices),
        " (dual-frame)" if dual_frame else "",
    )

    sample_times = [idx / video_fps for idx in sample_indices]
    all_audio: np.ndarray | None = None
    if audio_enabled:
        logger.info(
            "Audio enabled: extracting log-mel features (sr=%d, n_mels=%d)",
            audio_sample_rate,
            audio_n_mels,
        )
        all_audio = extract_frame_audio_features(
            total_view_path,
            sample_times,
            audio_sample_rate,
            audio_n_mels,
        )

    frames:         list[np.ndarray] = []
    labels:         list[int]        = []
    weights:        list[float]      = []
    kept_positions: list[int]        = []
    skipped:        int              = 0

    iterable = (
        tqdm(sample_indices, desc="Sampling frames", unit="frame")
        if _TQDM_AVAILABLE
        else sample_indices
    )

    for sample_pos, idx in enumerate(iterable):
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
            label, weight = 0, 1.0
        elif closeup_match and not wide_match:
            label, weight = 1, 1.0
        elif keep_transitions:
            # Ambiguous transition (fade/dissolve): keep it so the model sees the
            # cut moment, hard-labelled by the nearer camera but down-weighted.
            label  = 0 if diff_to_total <= diff_to_closeup else 1
            weight = transition_weight
        else:
            skipped += 1
            continue

        if dual_frame:
            wide_resized  = cv2.resize(f_total,   _STORE_SIZE)
            close_resized = cv2.resize(f_closeup, _STORE_SIZE)
            frames.append(np.concatenate([wide_resized, close_resized], axis=2))
        else:
            frames.append(cv2.resize(f_total, _STORE_SIZE))

        labels.append(label)
        weights.append(weight)
        kept_positions.append(sample_pos)

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

    stacked_frames = np.stack(frames)
    audio = all_audio[kept_positions] if all_audio is not None else None
    if audio is not None:
        logger.info("Audio features ready: %d samples x %d mels", audio.shape[0], audio.shape[1])

    torch.save(
        {
            "frames":  torch.from_numpy(stacked_frames),
            "labels":  torch.tensor(labels, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "audio":   torch.from_numpy(audio) if audio is not None else None,
        },
        str(cache_path),
    )
    logger.info("Cached %d labelled frames to %s", len(frames), cache_path)

    _print_label_summary(labels)
    return SampleSet(stacked_frames, labels, weights, audio)


def _print_label_summary(labels: list[int]) -> None:
    total   = len(labels)
    n_wide  = labels.count(0)
    n_close = labels.count(1)
    print(
        f"Sampled {total} frames: "
        f"{n_wide} wide ({n_wide / total * 100:.1f}%), "
        f"{n_close} close-up ({n_close / total * 100:.1f}%)"
    )


def _subset(data: SampleSet, idx: np.ndarray) -> SampleSet:
    """Index a SampleSet, keeping frames/labels/weights/audio aligned."""
    return SampleSet(
        frames  = data.frames[idx],
        labels  = [data.labels[i] for i in idx],
        weights = [data.weights[i] for i in idx],
        audio   = data.audio[idx] if data.audio is not None else None,
    )


def concat_samplesets(sets: list[SampleSet]) -> SampleSet:
    """Concatenate SampleSets; audio is kept only if every set has it."""
    audio = (
        np.concatenate([s.audio for s in sets if s.audio is not None])
        if all(s.audio is not None for s in sets) else None
    )
    return SampleSet(
        frames  = np.concatenate([s.frames for s in sets]),
        labels  = [l for s in sets for l in s.labels],
        weights = [w for s in sets for w in s.weights],
        audio   = audio,
    )


def split_dataset(data: SampleSet, val_split: float = 0.2, seed: int = 42) -> tuple[SampleSet, SampleSet]:
    """Shuffle and split a SampleSet into train/val subsets."""
    n       = len(data)
    indices = np.random.default_rng(seed).permutation(n)
    cut     = int(n * (1 - val_split))
    return _subset(data, indices[:cut]), _subset(data, indices[cut:])


def concert_split(concerts: list[SampleSet], val_concert_idx: int = -1) -> tuple[SampleSet, SampleSet]:
    """Split by concert: train on all except one, validate on the held-out one."""
    n = len(concerts)
    if n < 2:
        raise ValueError(
            "concert_split() requires at least 2 concerts. "
            "Use split_dataset() for single-concert training."
        )
    idx = val_concert_idx % n
    train = concat_samplesets([c for i, c in enumerate(concerts) if i != idx])
    return train, concerts[idx]
