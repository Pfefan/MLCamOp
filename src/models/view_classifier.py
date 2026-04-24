"""CNN-based binary classifier: wide shot (0) vs close-up (1) per frame.

MobileNetV3-Small pretrained on ImageNet, fine-tuned on concert footage with:
- Differential learning rates (backbone lr/10..lr/100, head lr)
- Inverse-frequency class weights in CrossEntropyLoss
- Balanced accuracy as early-stopping metric
- On-the-fly augmentation, AMP, cosine LR, gradient clipping
"""

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

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ImageNet normalisation constants (MobileNetV3 expects these)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_INPUT_SIZE    = (224, 224)

# DataLoader workers.  Each worker is a forked process; a single contiguous
# np.ndarray of shape (N,H,W,C) is truly COW-shared (index reads don't touch
# Python refcounts).  A Python list[np.ndarray] is NOT COW-safe — refcount
# updates on each list element trigger page copies, effectively duplicating the
# dataset per worker.  Keep workers low on RAM-constrained systems.
_NUM_WORKERS = 2

# Temporal sliding window: the dataset yields K consecutive frame pairs per
# sample.  The model mean-pools comparison features across K steps, giving it
# 2-3 seconds of context about which camera has been "active" recently.
# K=1 disables temporal context and behaves like the original single-frame mode.
_TEMPORAL_WINDOW = 5

# Augmentation pipelines applied only during training.
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
    """Apply augmentation to a 3-ch, 6-ch, or K*6-ch normalised tensor.

    Spatial transforms are applied identically across all channels (so all
    frames in the temporal window are flipped/affined consistently).  Color
    transforms are applied independently to each 3-channel camera view.
    """
    n_ch = tensor.shape[0]
    if n_ch % 6 == 0:
        # Apply the same spatial transform to the full (K*6, H, W) tensor
        tensor = _SPATIAL_AUGMENT(tensor)
        K = n_ch // 6
        parts: list[torch.Tensor] = []
        for k in range(K):
            parts.append(_COLOR_AUGMENT(tensor[k * 6     : k * 6 + 3]))
            parts.append(_COLOR_AUGMENT(tensor[k * 6 + 3 : k * 6 + 6]))
        return torch.cat(parts, dim=0)
    # 3-ch single-camera fallback
    return _SPATIAL_AUGMENT(_COLOR_AUGMENT(tensor))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 frame to a normalised CHW float tensor.

    Handles both 3-channel (single camera) and 6-channel (dual camera) frames.
    For 6-channel, each 3-channel half is independently normalised.
    """
    resized = cv2.resize(frame, _INPUT_SIZE)
    if resized.ndim == 3 and resized.shape[2] == 6:
        # Dual-frame: channels 0-2 = wide BGR, 3-5 = closeup BGR
        wide_rgb   = cv2.cvtColor(resized[:, :, :3], cv2.COLOR_BGR2RGB)
        close_rgb  = cv2.cvtColor(resized[:, :, 3:], cv2.COLOR_BGR2RGB)
        wide_t     = torch.from_numpy(wide_rgb).permute(2, 0, 1).float() / 255.0
        close_t    = torch.from_numpy(close_rgb).permute(2, 0, 1).float() / 255.0
        wide_t     = (wide_t  - _IMAGENET_MEAN) / _IMAGENET_STD
        close_t    = (close_t - _IMAGENET_MEAN) / _IMAGENET_STD
        return torch.cat([wide_t, close_t], dim=0)  # shape: (6, 224, 224)
    else:
        rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def _build_mobilenet(num_classes: int) -> nn.Module:
    """Load pretrained MobileNetV3-Small for single-frame (3-ch) classification.

    Freezes all but the last three feature blocks for domain adaptation.
    """
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
    """Siamese architecture: shared MobileNetV3 backbone processes each camera
    independently, then a classification head compares the two feature vectors.

    Uses standard Siamese comparison features: concat, absolute difference, and
    element-wise product — giving the head explicit similarity/difference signals
    rather than forcing it to discover comparison from raw concatenation alone.

    The last 5 feature blocks are unfrozen; earlier blocks stay frozen to preserve
    generic ImageNet features and prevent overfitting to just 2-3 training concerts.
    """

    # Number of trailing feature blocks to unfreeze (out of 13 total)
    _UNFREEZE_BLOCKS = 5

    def __init__(self, num_classes: int = 2):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Shared feature extractor (standard 3-channel pretrained backbone)
        self.features = base.features
        self.avgpool  = base.avgpool

        # Freeze all blocks, then selectively unfreeze the last _UNFREEZE_BLOCKS.
        # Unfreezing everything caused massive overfitting: val loss 0.87 → 3.8+
        # across 39 epochs with 56K training samples from only 2 concerts.
        for param in self.features.parameters():
            param.requires_grad = False
        for block in self.features[-self._UNFREEZE_BLOCKS :]:
            for param in block.parameters():
                param.requires_grad = True

        # Classification head with Siamese comparison features:
        # [wide, close, |wide-close|, wide*close] → 576 * 4 = 2304 dims
        feat_dim = 576
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 4, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Run 3-channel input through shared backbone → feature vector."""
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K*6, H, W) → per-frame Siamese comparison → temporal mean → classify.

        K=1 (single frame pair): standard Siamese compare.
        K>1 (temporal window):   compare each frame pair, mean-pool across time,
                                  then classify — giving the head 2-3 s of context.
        """
        K = x.shape[1] // 6
        comparisons: list[torch.Tensor] = []
        for k in range(K):
            wide_feat  = self._extract(x[:, k * 6     : k * 6 + 3])
            close_feat = self._extract(x[:, k * 6 + 3 : k * 6 + 6])
            comparisons.append(torch.cat([
                wide_feat,
                close_feat,
                torch.abs(wide_feat - close_feat),
                wide_feat * close_feat,
            ], dim=1))
        # (B, K, 2304) → mean over temporal dim → (B, 2304)
        pooled = torch.stack(comparisons, dim=1).mean(dim=1)
        return self.head(pooled)


class _FrameDataset(torch.utils.data.Dataset):
    """Dataset that stores raw uint8 numpy frames and preprocesses on-the-fly.

    Frames MUST be passed as a single contiguous np.ndarray of shape
    (N, H, W, C) — NOT as a list[np.ndarray].  A contiguous array is
    truly copy-on-write safe across DataLoader workers because indexing
    returns a view without touching Python reference counts.  A Python
    list would trigger COW page-copies on every worker access.

    When _TEMPORAL_WINDOW > 1, each sample is a window of K consecutive
    frame pairs (K*6 channels), centred on the labelled index.  Boundary
    indices are clamped so the first and last samples are padded by edge
    repetition.
    """

    def __init__(
        self,
        frames: np.ndarray,
        labels: list[int],
        augment: bool = False,
    ):
        if not isinstance(frames, np.ndarray):
            frames = np.asarray(frames)
        self.frames  = frames
        self.labels  = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor = self._build_window(idx)
        if self.augment:
            tensor = _augment_tensor(tensor)
        return tensor, self.labels[idx]

    def _build_window(self, idx: int) -> torch.Tensor:
        """Return K preprocessed frames centred on *idx* as a (K*6, H, W) tensor."""
        n    = len(self.frames)
        K    = _TEMPORAL_WINDOW
        half = K // 2
        indices = [max(0, min(n - 1, idx + i - half)) for i in range(K)]
        return torch.cat([_preprocess_frame(self.frames[j]) for j in indices], dim=0)


def _make_class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights: w_k = n_total / (K * n_k)."""
    counts  = torch.bincount(labels).float()
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device)


def _wrap_tqdm(iterable, **kwargs):
    """Return a tqdm-wrapped iterable if tqdm is installed, otherwise plain."""
    if _TQDM_AVAILABLE:
        return tqdm(iterable, **kwargs)
    return iterable


# ─────────────────────────────────────────────────────────────────────────────
# ViewClassifier
# ─────────────────────────────────────────────────────────────────────────────

class ViewClassifier:
    """Binary classifier: 0 = wide/total shot, 1 = close-up shot.

    Wraps MobileNetV3-Small pretrained on ImageNet, fine-tuned on concert data.
    """

    def __init__(self, num_classes: int = 2, dual_frame: bool = True):
        self.num_classes = num_classes
        self.dual_frame  = dual_frame
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = self._make_model().to(self.device)
        self.is_trained  = False

    def _make_model(self) -> nn.Module:
        """Build the appropriate architecture."""
        if self.dual_frame:
            return _DualStreamModel(self.num_classes)
        return _build_mobilenet(self.num_classes)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_frames: np.ndarray,
        train_labels: list[int],
        val_frames:   np.ndarray,
        val_labels:   list[int],
        epochs:       int   = 40,
        batch_size:   int   = 32,
        lr:           float = 3e-4,
        patience:     int   = 15,
        num_workers:  int   = _NUM_WORKERS,
    ) -> None:
        """Fine-tune with mixed-precision, early stopping on balanced accuracy."""
        train_ds = _FrameDataset(train_frames, train_labels, augment=True)
        val_ds   = _FrameDataset(val_frames,   val_labels,   augment=False)

        # pin_memory speeds up CPU→GPU transfers but requires extra page-locked
        # RAM.  Disable it when workers=0 (no async prefetch anyway) or when
        # the dataset is large and RAM is tight.
        pin = self.device.type == "cuda" and num_workers > 0
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=num_workers > 0,
        )

        y_train = torch.tensor(train_labels, dtype=torch.long)
        class_weights = _make_class_weights(y_train, self.device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        # Differential learning rates — 3 tiers across the unfrozen backbone blocks.
        # Works for both architectures:
        #   Single-frame: model.features + model.classifier
        #   Dual-stream:  model.features (shared) + model.head
        features = self.model.features

        if self.dual_frame:
            head_params = list(self.model.head.parameters())
            unfreeze = self.model._UNFREEZE_BLOCKS
            # 3 tiers across the unfrozen blocks: deep → mid → top
            deep_params = [p for b in self.model.features[-unfreeze    :-unfreeze//2] for p in b.parameters() if p.requires_grad]
            mid_params  = [p for b in self.model.features[-unfreeze//2 :-2          ] for p in b.parameters() if p.requires_grad]
            top_params  = [p for b in self.model.features[-2:                       ] for p in b.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": deep_params, "lr": lr / 100},
                    {"params": mid_params,  "lr": lr / 10},
                    {"params": top_params,  "lr": lr / 5},
                    {"params": head_params, "lr": lr},
                ],
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

        # CosineAnnealingLR: smooth decay from lr → eta_min over full budget.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # Mixed-precision scaler — no-op on CPU
        use_amp = self.device.type == "cuda"
        scaler  = GradScaler("cuda", enabled=use_amp)

        n_trainable = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        n_head = sum(p.numel() for p in head_params)
        counts = torch.bincount(y_train)
        print(f"Training on {len(train_frames)} | Validating on {len(val_frames)}")
        print(f"Trainable parameters: {n_trainable:,}  (backbone: {n_trainable - n_head:,} | head: {n_head:,})")
        print(f"Param groups: {' | '.join(f'{sum(p.numel() for p in g['params']):,} @ lr={g['lr']:.2e}' for g in optimizer.param_groups)}")
        print(f"Train class counts — wide: {counts[0].item()}  close-up: {counts[1].item()}")
        print(f"Class weights — wide: {class_weights[0]:.3f}  close-up: {class_weights[1]:.3f}")
        print(f"Device: {self.device} | AMP: {use_amp} | Workers: {num_workers}")

        best_bal_acc: float           = -1.0
        best_state:   Optional[dict]  = None
        no_improve:   int             = 0

        for epoch in range(1, epochs + 1):
            train_loss                    = self._run_epoch(train_loader, criterion, optimizer, scaler, training=True)
            val_loss, val_acc, bal_acc    = self._evaluate(val_loader, criterion)
            scheduler.step()

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"Acc: {val_acc * 100:.1f}% | "
                f"BalAcc: {bal_acc * 100:.1f}%"
            )

            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model — balanced accuracy: {best_bal_acc * 100:.1f}%")

        self.is_trained = True

    def _run_epoch(
        self,
        loader:    DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler:    GradScaler,
        training:  bool,
    ) -> float:
        """Run one full pass over *loader*, return mean loss."""
        self.model.train(training)
        total_loss = 0.0

        with torch.set_grad_enabled(training):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    outputs = self.model(batch_x)
                    loss    = criterion(outputs, batch_y)

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, float]:
        """Return (mean_loss, accuracy, balanced_accuracy)."""
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    outputs    = self.model(batch_x)
                    total_loss += criterion(outputs, batch_y).item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())

        # Standard accuracy
        correct  = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels)

        # Balanced accuracy: mean recall per class
        preds_t  = torch.tensor(all_preds)
        labels_t = torch.tensor(all_labels)
        n_classes = labels_t.max().item() + 1
        per_class_recall = []
        for c in range(n_classes):
            mask    = labels_t == c
            if mask.sum() > 0:
                recall = (preds_t[mask] == c).float().mean().item()
                per_class_recall.append(recall)
        bal_acc = sum(per_class_recall) / len(per_class_recall)

        return total_loss / len(loader), accuracy, bal_acc

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> int:
        """Predict for a single frame, repeating it K times as temporal context.

        For proper temporal context pass consecutive frames via predict_batch().
        """
        frames = np.stack([frame] * _TEMPORAL_WINDOW)
        return self.predict_batch(frames)[_TEMPORAL_WINDOW // 2]

    def predict_batch(self, frames: np.ndarray | list[np.ndarray], batch_size: int = 128) -> list[int]:
        """Predict labels for a sequence of frames with temporal context.

        Frames should be passed as one contiguous array of consecutive frames so
        that temporal windows are built correctly.  Each index i gets a window of
        the _TEMPORAL_WINDOW frames centred on i (edge-clamped).
        """
        self.model.eval()
        all_preds: list[int] = []

        if not isinstance(frames, np.ndarray):
            frames = np.asarray(frames)
        n    = len(frames)
        K    = _TEMPORAL_WINDOW
        half = K // 2

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            windows = []
            for idx in range(start, end):
                indices = [max(0, min(n - 1, idx + i - half)) for i in range(K)]
                windows.append(torch.cat([_preprocess_frame(frames[j]) for j in indices], dim=0))
            x = torch.stack(windows).to(self.device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = self.model(x)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

        return all_preds

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights to disk. Creates parent directories if needed."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "dual_frame": self.dual_frame},
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights from disk.

        Automatically detects dual_frame from checkpoint and rebuilds
        the architecture if needed.
        """
        data = torch.load(path, map_location=self.device, weights_only=False)

        # Support both old (bare state_dict) and new (dict with metadata) formats
        if isinstance(data, dict) and "state_dict" in data:
            state      = data["state_dict"]
            dual_frame = data.get("dual_frame", False)
        else:
            state      = data
            dual_frame = False

        # Rebuild model if dual_frame flag doesn't match
        if dual_frame != self.dual_frame:
            self.dual_frame = dual_frame
            self.model      = self._make_model().to(self.device)

        self.model.load_state_dict(state)
        self.is_trained = True
        print(f"Model loaded from {path}")
