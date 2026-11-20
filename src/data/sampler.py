"""Generate labelled training data from the wide, close-up, and result videos."""

import hashlib
import logging
import os
from dataclasses import dataclass, replace
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

_THUMB_W, _THUMB_H = 160, 90
_STORE_SIZE = (224, 224)


@dataclass
class SampleSet:
    """Per-sample training arrays that must stay index-aligned through splitting.

    frames:  (N, C, H, W) uint8 RGB frames (C=6 dual: wide + close-up stacked).
    labels:  hard nearest-camera class per frame (0 = wide, 1 = close-up).
    weights: per-sample loss weight; transition frames get < 1 (soft/uncertain).
    audio:   (N, feature_dim) pooled audio features, or None when audio is off.
    """

    frames:  np.ndarray
    labels:  list[int]
    weights: list[float]
    audio:   np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.frames)


def _thumb(frame: np.ndarray) -> np.ndarray:
    """Resize frame to thumbnail and convert to float32."""
    return cv2.resize(frame, (_THUMB_W, _THUMB_H)).astype(np.float32)


def _pixel_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Mean absolute pixel difference between two BGR frames (thumbnail scale)."""
    return float(np.mean(np.abs(_thumb(frame1) - _thumb(frame2))))


def _ncc_dist(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """1 - normalised cross-correlation of grayscale thumbnails.

    0 = identical, ~1 = uncorrelated, 2 = inverted. Invariant to global
    brightness/contrast shifts (colour grading, encoding differences), so one
    threshold works across concerts — unlike raw pixel diff.
    """
    a = cv2.cvtColor(cv2.resize(frame1, (_THUMB_W, _THUMB_H)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(frame2, (_THUMB_W, _THUMB_H)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-6:
        return 1.0
    return 1.0 - float((a * b).sum()) / denom


def _diff_fn(metric: str):
    """Resolve the frame-similarity metric by name."""
    if metric == "pixel":
        return _pixel_diff
    if metric == "ncc":
        return _ncc_dist
    raise ValueError(f"Unknown similarity metric: {metric!r} (expected 'pixel' or 'ncc')")


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    return cap


def _cache_key(total_view_path: str, closeup_path: str, result_path: str,
               sample_fps: float, threshold: float) -> str:
    """Hash of paths + settings so the cache is invalidated on any change."""
    payload = f"{total_view_path}|{closeup_path}|{result_path}|{sample_fps}|{threshold}"
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def compute_similarity_stats(
    total_view_path: str,
    closeup_path:    str,
    result_path:     str,
    n_samples:       int   = 200,
    metric:          str   = "pixel",
) -> dict:
    """Sample frames evenly across the videos and compute diff statistics for threshold tuning."""
    diff = _diff_fn(metric)
    cap_total   = _open_video(total_view_path)
    cap_closeup = _open_video(closeup_path)
    cap_result  = _open_video(result_path)

    total_frames = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))

    # Spread samples over the whole video — the opening minutes are often
    # wide-only, which skews the diff distributions.
    sample_indices = np.linspace(0, max(0, total_frames - 1), n_samples, dtype=int).tolist()

    diffs_total: list[float] = []
    diffs_closeup: list[float] = []

    for idx in sample_indices:
        for cap in (cap_total, cap_closeup, cap_result):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_t, f_total   = cap_total.read()
        ret_c, f_closeup = cap_closeup.read()
        ret_r, f_result  = cap_result.read()
        if ret_t and ret_c and ret_r:
            diffs_total.append(diff(f_result, f_total))
            diffs_closeup.append(diff(f_result, f_closeup))

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
    cache_id:             str | None = None,
    metric:               str   = "pixel",
) -> SampleSet:
    """Build a labelled sample set from the three concert videos."""
    diff = _diff_fn(metric)
    os.makedirs(cache_dir, exist_ok=True)
    key_parts = ["dual" if dual_frame else "single", metric]
    if keep_transitions:
        key_parts.append(f"tr{transition_weight}")
    if audio_enabled:
        key_parts.append(f"aud{audio_sample_rate}x{audio_n_mels}")
    if cache_id:
        settings = hashlib.md5(f"{sample_fps}|{similarity_threshold}|{metric}".encode()).hexdigest()[:6]
        base = f"{cache_id}_{settings}"
    else:
        base = _cache_key(total_view_path, closeup_path, result_path,
                          sample_fps, similarity_threshold)
    key = base + "_" + "_".join(key_parts)
    cache_path = Path(cache_dir) / f"frames_{key}.pt"

    if not force_rebuild and cache_path.exists():
        return _load_cache(cache_path)

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

    store_w, store_h = _STORE_SIZE
    n_channels = 6 if dual_frame else 3
    frame_buf = np.empty((len(sample_indices), n_channels, store_h, store_w), dtype=np.uint8)
    count:          int              = 0
    labels:         list[int]        = []
    weights:        list[float]      = []
    kept_positions: list[int]        = []
    skipped:        int              = 0

    # Sequential decode: grab() every frame (unavoidable for inter-frame codecs)
    # but only retrieve()/convert the sampled ones. This avoids the per-sample
    # keyframe seek that made random access an order of magnitude slower.
    progress = (
        tqdm(total=len(sample_indices), desc="Sampling frames", unit="frame")
        if _TQDM_AVAILABLE else None
    )

    idx = 0
    while idx < total_frames:
        if not (cap_total.grab() and cap_closeup.grab() and cap_result.grab()):
            break

        if idx % frame_interval == 0:
            ret_t, f_total   = cap_total.retrieve()
            ret_c, f_closeup = cap_closeup.retrieve()
            ret_r, f_result  = cap_result.retrieve()
            if progress is not None:
                progress.update(1)

            if ret_t and ret_c and ret_r:
                diff_to_total   = diff(f_result, f_total)
                diff_to_closeup = diff(f_result, f_closeup)

                wide_match    = diff_to_total   < similarity_threshold
                closeup_match = diff_to_closeup < similarity_threshold

                if wide_match and not closeup_match:
                    label, weight = 0, 1.0
                elif closeup_match and not wide_match:
                    label, weight = 1, 1.0
                elif keep_transitions:
                    # Ambiguous transition (fade/dissolve): keep it so the model
                    # sees the cut moment, hard-labelled by the nearer camera but
                    # down-weighted.
                    label, weight = (0 if diff_to_total <= diff_to_closeup else 1), transition_weight
                else:
                    label, weight = None, 0.0

                if label is not None:
                    if dual_frame:
                        frame_buf[count, :3] = cv2.cvtColor(
                            cv2.resize(f_total, _STORE_SIZE), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                        frame_buf[count, 3:] = cv2.cvtColor(
                            cv2.resize(f_closeup, _STORE_SIZE), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    else:
                        frame_buf[count] = cv2.cvtColor(
                            cv2.resize(f_total, _STORE_SIZE), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    labels.append(label)
                    weights.append(weight)
                    kept_positions.append(idx // frame_interval)
                    count += 1
                else:
                    skipped += 1
        idx += 1

    if progress is not None:
        progress.close()

    cap_total.release()
    cap_closeup.release()
    cap_result.release()

    if count == 0:
        raise RuntimeError(
            "No labelled frames generated.  Check video paths and run "
            "scripts/tune_threshold.py to calibrate similarity_threshold."
        )

    if skipped > 0:
        logger.info("Skipped %d ambiguous frames (transitions/fades)", skipped)

    stacked_frames = frame_buf[:count]
    audio = all_audio[kept_positions] if all_audio is not None else None
    if audio is not None:
        logger.info("Audio features ready: %d samples x %d dims", audio.shape[0], audio.shape[1])

    torch.save(
        {
            "frames":  torch.from_numpy(stacked_frames),
            "labels":  torch.tensor(labels, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "audio":   torch.from_numpy(audio) if audio is not None else None,
        },
        str(cache_path),
    )
    logger.info("Cached %d labelled frames to %s", count, cache_path)

    # Hand back the memory-mapped cache rather than the decoded buffer, so a
    # freshly built concert costs no more RAM than a cached one.
    return _load_cache(cache_path)


def _load_cache(cache_path: Path) -> SampleSet:
    """Memory-map a cached sample set."""
    logger.info("Loading cached training data from %s", cache_path)
    data    = torch.load(str(cache_path), weights_only=False, mmap=True)
    labels  = data["labels"].tolist()
    audio   = data["audio"].numpy() if data.get("audio") is not None else None
    _print_label_summary(labels)
    return SampleSet(data["frames"].numpy(), labels, data["weights"].tolist(), audio)


def load_concert(data_cfg: dict, concert: dict, force_rebuild: bool = False) -> SampleSet:
    """Build (or load from cache) one concert's sample set.

    *data_cfg* is the ``data`` section of the config and *concert* one entry of
    its ``concerts`` list; a concert may override ``similarity_threshold``.
    """
    labeling  = data_cfg.get("labeling", {})
    audio_cfg = data_cfg.get("audio", {})
    return generate_training_data(
        total_view_path      = concert["total_view"],
        closeup_path         = concert["closeup"],
        result_path          = concert["result"],
        sample_fps           = data_cfg.get("sample_fps", 1.0),
        similarity_threshold = concert.get("similarity_threshold",
                                           data_cfg.get("similarity_threshold", 39)),
        cache_dir            = data_cfg.get("cache_dir", "data/cache"),
        force_rebuild        = force_rebuild,
        dual_frame           = data_cfg.get("dual_frame", True),
        keep_transitions     = labeling.get("keep_transitions", True),
        transition_weight    = labeling.get("transition_weight", 0.3),
        audio_enabled        = audio_cfg.get("enabled", False),
        audio_sample_rate    = audio_cfg.get("sample_rate", 16000),
        audio_n_mels         = audio_cfg.get("n_mels", 32),
        cache_id             = concert.get("id"),
        metric               = data_cfg.get("similarity_metric", "pixel"),
    )


def concert_name(concert: dict, index: int) -> str:
    """Display name of a concert entry: its ``id``, else its 1-based position."""
    return concert.get("id") or f"concert {index + 1}"


def val_concert_indices(data_cfg: dict) -> list[int]:
    """Positions of the held-out concerts in ``data.concerts``.

    Concerts are held out by ``id`` through ``data.val_concerts``, so adding or
    reordering concerts never changes the validation set. Without that key the
    last concert is held out.
    """
    concerts = data_cfg["concerts"]
    val_ids  = data_cfg.get("val_concerts")
    if not val_ids:
        return [len(concerts) - 1]

    position = {concert.get("id"): i for i, concert in enumerate(concerts)}
    unknown  = [val_id for val_id in val_ids if val_id not in position]
    if unknown:
        raise ValueError(f"data.val_concerts lists unknown concert id(s): {unknown}")
    if len(set(val_ids)) >= len(concerts):
        raise ValueError("data.val_concerts holds out every concert; nothing left to train on.")
    return sorted(position[val_id] for val_id in set(val_ids))


def balance_concert_weights(concerts: list[SampleSet]) -> list[SampleSet]:
    """Rescale sample weights so every (concert, class) pair carries equal total weight.

    Equal class mass inside each concert stops the model from using the venue as a
    shortcut for that concert's wide/close-up ratio; equal mass per concert stops
    long concerts from dominating. Relative weights inside a pair (down-weighted
    transitions) are kept and the mean weight stays 1. For a single concert this
    is plain inverse-frequency class weighting.
    """
    labels  = [np.asarray(concert.labels) for concert in concerts]
    weights = [np.asarray(concert.weights, dtype=np.float64) for concert in concerts]
    masses  = [
        {cls: concert_weights[concert_labels == cls].sum() for cls in np.unique(concert_labels)}
        for concert_labels, concert_weights in zip(labels, weights)
    ]
    n_pairs = sum(1 for concert_masses in masses for mass in concert_masses.values() if mass > 0)
    target  = sum(len(concert_labels) for concert_labels in labels) / max(n_pairs, 1)

    balanced: list[SampleSet] = []
    for concert, concert_labels, concert_weights, concert_masses in zip(concerts, labels, weights, masses):
        scale = np.zeros(len(concert_labels))
        for cls, mass in concert_masses.items():
            if mass > 0:
                scale[concert_labels == cls] = target / mass
        balanced.append(replace(concert, weights=(concert_weights * scale).tolist()))
    return balanced


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
