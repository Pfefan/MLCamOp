"""Per-frame audio features (log-mel) aligned to sampled video timestamps.

The mel transform is pure torch; only decoding the audio track from the video
container needs torchaudio's I/O backend (torchcodec + ffmpeg). Decode failures
raise an actionable error so the caller can install the backend or disable audio.
"""

import numpy as np
import torch

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

_HOP_LENGTH = 512


def _load_mono(video_path: str, target_sr: int) -> torch.Tensor:
    """Decode the audio track to a mono waveform at `target_sr`."""
    if not _TORCHAUDIO_AVAILABLE:
        raise RuntimeError(
            "Audio branch enabled but torchaudio is not installed. "
            "Install it or set data.audio.enabled: false."
        )
    try:
        waveform, sr = torchaudio.load(video_path)
    except Exception as exc:  # decode backend (torchcodec/ffmpeg) missing or no track
        raise RuntimeError(
            f"Failed to decode audio from {video_path}: {exc}. "
            "torchaudio audio decode needs torchcodec + ffmpeg; install them, "
            "or set data.audio.enabled: false."
        ) from exc

    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def extract_frame_audio_features(
    video_path:  str,
    timestamps:  list[float],
    sample_rate: int = 16000,
    n_mels:      int = 32,
) -> np.ndarray:
    """Return an (N, n_mels) log-mel feature — one vector per sampled timestamp.

    The waveform is turned into a log-mel spectrogram over time; each timestamp
    picks its nearest mel frame, giving one audio feature per video sample.
    """
    waveform = _load_mono(video_path, sample_rate)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, hop_length=_HOP_LENGTH,
    )(waveform)
    log_mel = torch.log1p(mel).squeeze(0).transpose(0, 1)  # (frames, n_mels)

    frame_times = torch.arange(log_mel.shape[0]) * _HOP_LENGTH / sample_rate
    features = [
        log_mel[int(torch.argmin(torch.abs(frame_times - ts)))]
        for ts in timestamps
    ]
    return torch.stack(features).numpy().astype(np.float32)
