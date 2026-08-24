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


def extract_frame_audio_features(
    video_path:  str,
    timestamps:  list[float],
    sample_rate: int = 16000,
    n_mels:      int = 32,
) -> np.ndarray:
    """Return one log-mel feature vector per sampled timestamp."""
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

    nearest    = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    take_left  = np.abs(query - times[nearest - 1]) <= np.abs(times[nearest] - query)
    nearest    = np.where(take_left, nearest - 1, nearest)
    return features[nearest].astype(np.float32)
