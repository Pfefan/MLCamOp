"""Audio features aligned to sampled video timestamps."""

import numpy as np
import torch

try:
    import torchaudio
    from torchcodec.decoders import AudioDecoder
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

_HOP_LENGTH    = 512
_CHUNK_SECONDS = 300
_POOL_AGG      = 3  # mean, std, max per mel band


def frame_audio_feature_dim(n_mels: int) -> int:
    """Feature width per sampled frame (mean/std/max pooling of the mel bands)."""
    return n_mels * _POOL_AGG


def extract_frame_audio_features(
    video_path:  str,
    timestamps:  list[float],
    sample_rate: int = 16000,
    n_mels:      int = 32,
) -> np.ndarray:
    """Return one pooled, per-clip-standardised audio vector per sampled timestamp."""
    if not _AUDIO_AVAILABLE:
        raise RuntimeError(
            "Audio branch enabled but torchaudio/torchcodec are not installed. "
            "Install them (and ffmpeg) or set data.audio.enabled: false."
        )
    try:
        decoder = AudioDecoder(video_path, sample_rate=sample_rate, num_channels=1)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open audio from {video_path}: {exc}. "
            "torchcodec audio decode needs ffmpeg; install it, or set "
            "data.audio.enabled: false."
        ) from exc

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, hop_length=_HOP_LENGTH,
    )
    duration = float(decoder.metadata.duration_seconds or 0.0)

    log_mels:    list[torch.Tensor] = []
    frame_times: list[np.ndarray]   = []
    start = 0.0
    while start < duration:
        stop     = min(start + _CHUNK_SECONDS, duration)
        waveform = decoder.get_samples_played_in_range(start, stop).data
        if waveform.shape[1] >= _HOP_LENGTH:
            mel = mel_transform(waveform)
            log_mel = torch.log1p(mel).squeeze(0).transpose(0, 1)
            log_mels.append(log_mel)
            frame_times.append(start + np.arange(log_mel.shape[0]) * _HOP_LENGTH / sample_rate)
        start = stop

    if not log_mels:
        raise RuntimeError(f"No audio decoded from {video_path}.")

    features   = torch.cat(log_mels, dim=0).numpy()          # (F, n_mels)
    times      = np.concatenate(frame_times)                 # (F,) monotonic
    query      = np.asarray(timestamps, dtype=np.float64)

    return _pool_and_standardize(features, times, query)


def _pool_and_standardize(mel: np.ndarray, times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Pool every mel frame inside each sample's slot, then z-score per clip.

    Video is sampled sparsely (e.g. every 0.5 s) but audio is continuous, so each
    sampled timestamp aggregates mean/std/max over *all* mel frames in its slot
    instead of a single instant — no audio between samples is discarded.
    Standardising per clip removes concert-to-concert loudness differences.
    """
    n = len(query)
    m = mel.shape[1]
    if n == 0 or len(times) == 0:
        return np.zeros((n, m * _POOL_AGG), dtype=np.float32)

    # Assign each mel frame to its nearest sample → contiguous, sorted partition.
    mids   = (query[:-1] + query[1:]) / 2.0 if n > 1 else np.array([np.inf])
    bucket = np.searchsorted(mids, times)                    # (F,) in [0, n-1]

    present = np.unique(bucket)
    starts  = np.searchsorted(bucket, present, side="left")
    counts  = np.diff(np.append(starts, len(times)))[:, None]

    s    = np.add.reduceat(mel,       starts, axis=0)
    s2   = np.add.reduceat(mel * mel, starts, axis=0)
    mx   = np.maximum.reduceat(mel,   starts, axis=0)
    mean = s / counts
    std  = np.sqrt(np.clip(s2 / counts - mean * mean, 0.0, None))
    feat_present = np.concatenate([mean, std, mx], axis=1)   # (P, 3M)

    # Scatter to every sample; empty slots borrow the nearest populated one.
    if len(present) == 1:
        feats = np.repeat(feat_present, n, axis=0)
    else:
        idx       = np.arange(n)
        pos       = np.clip(np.searchsorted(present, idx), 1, len(present) - 1)
        take_left = (idx - present[pos - 1]) <= (present[pos] - idx)
        nearest   = np.where(take_left, pos - 1, pos)
        feats     = feat_present[nearest]                    # (N, 3M)

    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    return ((feats - mu) / sd).astype(np.float32)
