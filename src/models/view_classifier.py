"""Wide-vs-close-up frame classifier."""

import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
import torchvision.transforms.v2 as T

from src.data.sampler import SampleSet, balance_concert_weights
from src.utils.metrics import balanced_accuracy, cut_event_prf, viterbi_smooth

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

logger = logging.getLogger("ConcertVideoEditor")

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_INPUT_SIZE = (224, 224)
_NUM_WORKERS = 2
_TEMPORAL_WINDOW = 5
_LOOKAHEAD = 1
_RNN_HIDDEN = 128


def _causal_window_indices(idx: int, n: int, window: int, lookahead: int) -> list[int]:
    """Return edge-clamped indices for a causal window."""
    start = idx - (window - 1 - lookahead)
    return [min(n - 1, max(0, start + i)) for i in range(window)]

_SPATIAL_AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

_COLOR_AUGMENT = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    T.RandomGrayscale(p=0.1),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
])


def _augment_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Apply training-time image augmentation."""
    n_ch = tensor.shape[0]
    if n_ch % 6 == 0:
        tensor = _SPATIAL_AUGMENT(tensor)
        K = n_ch // 6
        h, w = tensor.shape[-2:]
        views = tensor.reshape(K, 2, 3, h, w)
        # One jitter draw per camera, shared across timesteps.
        wide  = _COLOR_AUGMENT(views[:, 0])
        close = _COLOR_AUGMENT(views[:, 1])
        return torch.stack([wide, close], dim=1).reshape(n_ch, h, w)
    return _SPATIAL_AUGMENT(_COLOR_AUGMENT(tensor))


def _augment_batch(windows: torch.Tensor) -> torch.Tensor:
    """Augment every window of a batch with its own random draw, on the batch's device."""
    return torch.stack([_augment_tensor(window) for window in windows])


def _preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Convert a BGR frame to a normalised CHW tensor."""
    if frame.shape[1] != _INPUT_SIZE[0] or frame.shape[0] != _INPUT_SIZE[1]:
        resized = cv2.resize(frame, _INPUT_SIZE)
    else:
        resized = frame
    if resized.ndim == 3 and resized.shape[2] == 6:
        wide_rgb = cv2.cvtColor(resized[:, :, :3], cv2.COLOR_BGR2RGB)
        close_rgb = cv2.cvtColor(resized[:, :, 3:], cv2.COLOR_BGR2RGB)
        wide_t = torch.from_numpy(wide_rgb).permute(2, 0, 1).float() / 255.0
        close_t = torch.from_numpy(close_rgb).permute(2, 0, 1).float() / 255.0
        wide_t = (wide_t - _IMAGENET_MEAN) / _IMAGENET_STD
        close_t = (close_t - _IMAGENET_MEAN) / _IMAGENET_STD
        return torch.cat([wide_t, close_t], dim=0)
    else:
        rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def _frame_to_uint8_chw(frame: np.ndarray) -> torch.Tensor:
    """Convert a frame to an RGB uint8 CHW tensor (no-op for cached CHW frames)."""
    if frame.ndim == 3 and frame.shape[0] in (3, 6):
        return torch.from_numpy(np.ascontiguousarray(frame))
    if frame.shape[1] != _INPUT_SIZE[0] or frame.shape[0] != _INPUT_SIZE[1]:
        frame = cv2.resize(frame, _INPUT_SIZE)
    if frame.ndim == 3 and frame.shape[2] == 6:
        order = [2, 1, 0, 5, 4, 3]
    else:
        order = [2, 1, 0]
    rgb = np.ascontiguousarray(frame[..., order])
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _normalize_windows(x: torch.Tensor) -> torch.Tensor:
    """Convert uint8 RGB windows to ImageNet-normalised float tensors."""
    c = x.shape[1]
    mean = _IMAGENET_MEAN.to(x.device).repeat(c // 3, 1, 1)
    std  = _IMAGENET_STD.to(x.device).repeat(c // 3, 1, 1)
    x    = x.float().div_(255.0).sub_(mean).div_(std)
    return x.contiguous(memory_format=torch.channels_last)


def _build_mobilenet(num_classes: int) -> nn.Module:
    """Load a pretrained MobileNetV3-Small and adapt the head."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False
    for block in model.features[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.5),
        nn.Linear(128, num_classes),
    )
    return model


class _DualStreamModel(nn.Module):
    """Shared-backbone temporal model over a causal window."""

    # Learning-rate multiplier per unfrozen backbone block, deepest block first.
    _UNFROZEN_LR_SCALES = (0.01, 0.1, 0.2)
    _UNFREEZE_BLOCKS = len(_UNFROZEN_LR_SCALES)
    _REDUCED_CHANNELS = 128
    _COMPARISON_DIM = _REDUCED_CHANNELS * 4

    def __init__(self, num_classes: int = 2, rnn_hidden: int = _RNN_HIDDEN, audio_dim: int = 0):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        self.features = base.features
        self.avgpool  = base.avgpool
        self.reduce = nn.Sequential(
            nn.Conv2d(576, self._REDUCED_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(self._REDUCED_CHANNELS),
            nn.Hardswish(),
        )

        for param in self.features.parameters():
            param.requires_grad = False
        for block in self.features[-self._UNFREEZE_BLOCKS :]:
            for param in block.parameters():
                param.requires_grad = True

        self.temporal = nn.GRU(
            input_size=self._COMPARISON_DIM + audio_dim,
            hidden_size=rnn_hidden,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def train(self, mode: bool = True) -> "_DualStreamModel":
        """Switch train/eval mode while keeping the frozen blocks in eval mode.

        Frozen blocks would otherwise re-estimate their BatchNorm statistics from
        the training concerts and drift away from the pretrained weights they feed.
        """
        super().train(mode)
        for block in self.features[: -self._UNFREEZE_BLOCKS]:
            block.eval()
        return self

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Run camera frames through the shared backbone."""
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.features(x)
        x = self.reduce(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor, audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run a window through the dual-stream model."""
        batch, channels, height, width = x.shape
        steps = channels // 6
        # A single backbone pass over both cameras keeps the BatchNorm batch
        # statistics mixed, like the running statistics used at inference.
        feats = self._extract(x.reshape(batch * steps * 2, 3, height, width))
        feats = feats.reshape(batch, steps, 2, -1)
        wide_feat, close_feat = feats[:, :, 0], feats[:, :, 1]

        seq = torch.cat([
            wide_feat,
            close_feat,
            torch.abs(wide_feat - close_feat),
            wide_feat * close_feat,
        ], dim=2)                              # (B, K, comparison_dim)
        if audio is not None:
            seq = torch.cat([seq, audio], dim=2)

        _, hidden = self.temporal(seq)         # hidden: (1, B, rnn_hidden)
        return self.head(hidden[-1])


class _FrameDataset(torch.utils.data.Dataset):
    """Dataset that yields causal temporal windows for a sample set."""

    def __init__(
        self,
        frames: np.ndarray,
        labels: list[int],
        window: int = _TEMPORAL_WINDOW,
        lookahead: int = _LOOKAHEAD,
        weights: list[float] | None = None,
        audio: np.ndarray | None = None,
    ):
        self.window    = window
        self.lookahead = lookahead
        if not isinstance(frames, np.ndarray):
            frames = np.asarray(frames)
        self._concerts    = [(frames, labels, weights, audio)]
        self._legacy_pair = weights is None and audio is None
        self._build_index()

    @classmethod
    def from_concerts(cls, concerts: list, window: int, lookahead: int) -> "_FrameDataset":
        self = cls.__new__(cls)
        self.window    = window
        self.lookahead = lookahead
        self._concerts = [(c.frames, c.labels, c.weights, c.audio) for c in concerts]
        self._legacy_pair = False
        self._build_index()
        return self

    def _build_index(self) -> None:
        self._index = [
            (ci, li)
            for ci, (frames, *_rest) in enumerate(self._concerts)
            for li in range(len(frames))
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, gidx: int):
        ci, li = self._index[gidx]
        frames, labels, weights, audio = self._concerts[ci]

        tensor = self._build_window(frames, li)

        if self._legacy_pair:
            return tensor, labels[li]

        weight     = weights[li] if weights is not None else 1.0
        audio_win  = self._audio_window(audio, li) if audio is not None else torch.zeros(0)
        return tensor, labels[li], weight, audio_win

    def _audio_window(self, audio: np.ndarray, li: int) -> torch.Tensor:
        """Audio features for the same causal window."""
        indices = _causal_window_indices(li, len(audio), self.window, self.lookahead)
        return torch.from_numpy(audio[indices]).float()

    def _build_window(self, frames: np.ndarray, li: int) -> torch.Tensor:
        """Build the causal window for a given frame index."""
        indices = _causal_window_indices(li, len(frames), self.window, self.lookahead)
        return torch.cat([_frame_to_uint8_chw(frames[j]) for j in indices], dim=0)


def _validation_scores(
    probs: np.ndarray,
    concert_labels: list[list[int]],
    smooth_penalty: float,
    smooth_min_shot: int,
    cut_tolerance_samples: int,
) -> tuple[float, float, float]:
    """Return (accuracy, balanced accuracy, smoothed cut-F1) for held-out concerts.

    *probs* is the concatenation of the concerts' probabilities in order. Balanced
    accuracy and cut-F1 are averaged per concert, so every held-out concert counts
    equally and smoothing never runs across a concert boundary.
    """
    bounds = np.cumsum([len(labels) for labels in concert_labels])[:-1]
    correct = 0
    bal_accs: list[float] = []
    cut_f1s:  list[float] = []
    for labels, concert_probs in zip(concert_labels, np.split(probs, bounds)):
        preds    = concert_probs.argmax(axis=1).tolist()
        smoothed = viterbi_smooth(concert_probs, smooth_penalty, min_shot=smooth_min_shot)
        correct += sum(p == l for p, l in zip(preds, labels))
        bal_accs.append(balanced_accuracy(labels, preds))
        cut_f1s.append(cut_event_prf(labels, smoothed, cut_tolerance_samples)["f1"])
    return correct / len(probs), float(np.mean(bal_accs)), float(np.mean(cut_f1s))


def _wrap_tqdm(iterable, **kwargs):
    """Wrap an iterable with tqdm when available."""
    if _TQDM_AVAILABLE:
        return tqdm(iterable, **kwargs)
    return iterable


class ViewClassifier:
    """Binary wide-vs-close-up classifier."""

    def __init__(
        self,
        num_classes: int = 2,
        dual_frame: bool = True,
        window: int = _TEMPORAL_WINDOW,
        lookahead: int = _LOOKAHEAD,
        rnn_hidden: int = _RNN_HIDDEN,
        audio_dim: int = 0,
        audio_sample_rate: int = 16000,
    ):
        self.num_classes = num_classes
        self.dual_frame  = dual_frame
        self.window      = window
        self.lookahead   = lookahead
        self.rnn_hidden  = rnn_hidden
        self.audio_dim   = audio_dim
        self.audio_sample_rate = audio_sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._make_model().to(
            self.device, memory_format=torch.channels_last
        )
        self._model_fwd = self.model
        self.is_trained = False

    def _make_model(self) -> nn.Module:
        """Build the model for the selected input mode."""
        if self.dual_frame:
            return _DualStreamModel(self.num_classes, rnn_hidden=self.rnn_hidden, audio_dim=self.audio_dim)
        return _build_mobilenet(self.num_classes)

    def _forward(self, x: torch.Tensor, audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run one forward pass, with audio only for the dual-stream model."""
        if self.dual_frame:
            return self._model_fwd(x, audio)
        return self._model_fwd(x)

    def train(
        self,
        train_concerts: list[SampleSet],
        val_concerts: list[SampleSet],
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 3e-4,
        patience: int = 15,
        num_workers: int = _NUM_WORKERS,
        use_compile: bool = True,
        cut_tolerance_samples: int = 2,
        smooth_penalty: float = 0.1,
        cutf1_checkpoint: Optional[str] = None,
        label_smoothing: float = 0.1,
        smooth_min_shot: int = 1,
        prefetch_factor: int = 2,
    ) -> None:
        """Train the model with early stopping on balanced accuracy and cut-F1.

        Both metrics are averaged over *val_concerts*, the held-out concerts.
        """
        train_labels = [l for c in train_concerts for l in c.labels]
        n_train      = sum(len(c.frames) for c in train_concerts)
        n_val        = sum(len(c.frames) for c in val_concerts)

        balanced_train = balance_concert_weights(train_concerts)
        train_ds = _FrameDataset.from_concerts(
            balanced_train, window=self.window, lookahead=self.lookahead,
        )
        val_ds   = _FrameDataset.from_concerts(
            val_concerts, window=self.window, lookahead=self.lookahead,
        )

        pin = self.device.type == "cuda" and num_workers > 0
        prefetch = prefetch_factor if num_workers > 0 else None
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
        )

        self._model_fwd = self.model
        env_disabled = os.environ.get("MLCAMOP_NO_COMPILE") == "1"
        if self.device.type == "cuda" and use_compile and not env_disabled:
            try:
                self._model_fwd = torch.compile(self.model)
                logger.info("torch.compile: enabled")
            except Exception as exc:  # pragma: no cover - depends on backend
                logger.info("torch.compile unavailable (%s); running eager", exc)

        criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

        features = self.model.features

        if self.dual_frame:
            head_params = (
                list(self.model.head.parameters())
                + list(self.model.temporal.parameters())
                + list(self.model.reduce.parameters())
            )
            unfreeze = self.model._UNFREEZE_BLOCKS
            backbone_groups = [
                {"params": list(block.parameters()), "lr": lr * scale}
                for block, scale in zip(features[-unfreeze:], self.model._UNFROZEN_LR_SCALES)
            ]
            optimizer = torch.optim.AdamW(
                [*backbone_groups, {"params": head_params, "lr": lr}],
                weight_decay=5e-3,
            )
        else:
            backbone_params_deep = [
                p for p in features[-3].parameters() if p.requires_grad
            ]
            backbone_params_mid = [
                p for p in features[-2].parameters() if p.requires_grad
            ]
            backbone_params_top = [
                p for p in features[-1].parameters() if p.requires_grad
            ]
            head_params = list(self.model.classifier.parameters())
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params_deep, "lr": lr / 100},
                    {"params": backbone_params_mid,  "lr": lr / 10},
                    {"params": backbone_params_top,  "lr": lr / 5},
                    {"params": head_params,          "lr": lr},
                ],
                weight_decay=1e-3,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        use_amp = self.device.type == "cuda"
        scaler  = GradScaler("cuda", enabled=use_amp)

        n_trainable = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        n_head = sum(p.numel() for p in head_params)
        sample_weights = [w for c in balanced_train for w in c.weights]
        logger.info("Training on %d | Validating on %d", n_train, n_val)
        logger.info(
            "Trainable parameters: %s  (backbone: %s | head: %s)",
            f"{n_trainable:,}", f"{n_trainable - n_head:,}", f"{n_head:,}",
        )
        logger.info("Param groups: %s", " | ".join(
            f"{sum(p.numel() for p in g['params']):,} @ lr={g['lr']:.2e}" for g in optimizer.param_groups
        ))
        logger.info(
            "Train class counts — wide: %d  close-up: %d",
            train_labels.count(0), train_labels.count(1),
        )
        logger.info(
            "Concert-balanced sample weights — min: %.2f  max: %.2f",
            min(sample_weights), max(sample_weights),
        )
        logger.info("Device: %s | AMP: %s | Workers: %d", self.device, use_amp, num_workers)

        best_bal_acc: float           = -1.0
        best_state:   Optional[dict]  = None
        best_cut_f1:  float           = -1.0
        best_cut_state: Optional[dict] = None
        no_improve:   int             = 0
        val_labels                    = [list(c.labels) for c in val_concerts]

        for epoch in range(1, epochs + 1):
            train_loss      = self._run_epoch(train_loader, criterion, optimizer, scaler, training=True)
            val_loss, probs = self._evaluate(val_loader, criterion)
            scheduler.step()

            val_acc, bal_acc, cut_f1 = _validation_scores(
                probs, val_labels, smooth_penalty, smooth_min_shot, cut_tolerance_samples,
            )

            logger.info(
                "Epoch %02d/%d | Train: %.4f | Val: %.4f | Acc: %.1f%% | BalAcc: %.1f%% | CutF1: %.1f%%",
                epoch, epochs, train_loss, val_loss, val_acc * 100, bal_acc * 100, cut_f1 * 100,
            )

            improved_cut = cut_f1 > best_cut_f1
            improved_bal = bal_acc > best_bal_acc
            if improved_cut:
                best_cut_f1    = cut_f1
                best_cut_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            if improved_bal:
                best_bal_acc = bal_acc
                best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # Stop only once neither objective has improved for `patience` epochs.
            if improved_cut or improved_bal:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d) — neither BalAcc nor CutF1 improved",
                        epoch, patience,
                    )
                    break

        if best_cut_state is not None and cutf1_checkpoint:
            self.model.load_state_dict(best_cut_state)
            self.save(cutf1_checkpoint)
            logger.info("Saved best-by-cut-F1 model (%.1f%%) to %s", best_cut_f1 * 100, cutf1_checkpoint)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best model — balanced accuracy: %.1f%%", best_bal_acc * 100)

        self.is_trained = True

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        training: bool,
    ) -> float:
        """Run one epoch and return mean loss."""
        self.model.train(training)
        total_loss = 0.0

        with torch.set_grad_enabled(training):
            for batch_x, batch_y, batch_w, batch_a in loader:
                # Windows stay uint8 until they are on the device: 4× smaller transfer,
                # and augmentation there is far cheaper than in the loader workers.
                batch_x = batch_x.to(self.device, non_blocking=True)
                if training:
                    batch_x = _augment_batch(batch_x)
                batch_x = _normalize_windows(batch_x)
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_w = batch_w.to(self.device, non_blocking=True).float()
                audio   = batch_a.to(self.device).float() if batch_a.numel() else None

                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    outputs = self._forward(batch_x, audio)
                    loss = (criterion(outputs, batch_y) * batch_w).sum() / batch_w.sum()

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray]:
        """Return (loss, class probabilities in loader order)."""
        total_loss = 0.0
        all_probs: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_w, batch_a in loader:
                batch_x = _normalize_windows(batch_x.to(self.device, non_blocking=True))
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_w = batch_w.to(self.device, non_blocking=True).float()
                audio   = batch_a.to(self.device).float() if batch_a.numel() else None
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    outputs     = self._forward(batch_x, audio)
                    total_loss += ((criterion(outputs, batch_y) * batch_w).sum() / batch_w.sum()).item()
                all_probs.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())

        probs_arr = (
            np.concatenate(all_probs, axis=0) if all_probs
            else np.zeros((0, self.num_classes), dtype=np.float32)
        )
        return total_loss / len(loader), probs_arr

    def predict(self, frame: np.ndarray) -> int:
        """Predict a single frame using a repeated temporal window."""
        frames = np.stack([frame] * self.window)
        return self.predict_batch(frames)[self.window // 2]

    def predict_window(self, window_frames: list[np.ndarray], window_audio: np.ndarray | None = None) -> int:
        """Classify one causal window of frames."""
        return int(self.predict_window_proba(window_frames, window_audio).argmax())

    def predict_window_proba(self, window_frames: list[np.ndarray], window_audio: np.ndarray | None = None) -> np.ndarray:
        """Class probabilities for one causal window of frames."""
        x = torch.cat([_frame_to_uint8_chw(f) for f in window_frames], dim=0)
        x = _normalize_windows(x.unsqueeze(0).to(self.device, non_blocking=True))
        audio = None
        if window_audio is not None:
            audio = torch.from_numpy(window_audio).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self._forward(x, audio)
        return torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()

    def predict_batch(
        self,
        frames: np.ndarray | list[np.ndarray],
        audio: np.ndarray | None = None,
        batch_size: int = 128,
    ) -> list[int]:
        """Predict labels for a sequence of frames."""
        probs = self.predict_proba_batch(frames, audio=audio, batch_size=batch_size)
        return probs.argmax(axis=1).astype(int).tolist()

    def predict_proba_batch(
        self,
        frames: np.ndarray | list[np.ndarray],
        audio: np.ndarray | None = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Per-frame class probabilities (N, num_classes) for sequence smoothing."""
        self.model.eval()
        all_probs: list[np.ndarray] = []

        if not isinstance(frames, np.ndarray):
            frames = np.asarray(frames)
        n = len(frames)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            windows, audio_windows = [], []
            for idx in range(start, end):
                indices = _causal_window_indices(idx, n, self.window, self.lookahead)
                windows.append(torch.cat([_frame_to_uint8_chw(frames[j]) for j in indices], dim=0))
                if audio is not None:
                    audio_windows.append(torch.from_numpy(audio[indices]).float())
            x = _normalize_windows(torch.stack(windows).to(self.device, non_blocking=True))
            a = torch.stack(audio_windows).to(self.device) if audio is not None else None
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = self._forward(x, a)
            probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "dual_frame": self.dual_frame,
                "num_classes": self.num_classes,
                "window": self.window,
                "lookahead": self.lookahead,
                "rnn_hidden": self.rnn_hidden,
                "audio_dim": self.audio_dim,
                "audio_sample_rate": self.audio_sample_rate,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights from a checkpoint."""
        data = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(data, dict) and "state_dict" in data:
            state = data["state_dict"]
            self.dual_frame  = data.get("dual_frame", self.dual_frame)
            self.num_classes = data.get("num_classes", self.num_classes)
            self.window      = data.get("window", self.window)
            self.lookahead   = data.get("lookahead", self.lookahead)
            self.rnn_hidden  = data.get("rnn_hidden", self.rnn_hidden)
            self.audio_dim   = data.get("audio_dim", self.audio_dim)
            self.audio_sample_rate = data.get("audio_sample_rate", self.audio_sample_rate)
        else:
            state = data

        self.model = self._make_model().to(self.device, memory_format=torch.channels_last)
        self._model_fwd = self.model
        self.model.load_state_dict(state)
        self.is_trained = True
        print(f"Model loaded from {path}")
