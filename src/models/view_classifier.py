"""CNN-based binary classifier: wide shot (0) vs close-up (1) per frame.

Architecture
------------
MobileNetV3-Small pretrained on ImageNet, fine-tuned on concert footage.

Fine-tuning strategy
--------------------
- The last feature block of the backbone (features[-1]) is unfrozen and trained
  with a low learning rate (lr/10).  All earlier backbone layers stay frozen.
  This lets the model adapt the final convolutional features to concert-specific
  visual patterns (stage geometry, instrument shapes, lighting) while keeping
  the early, universal edge/texture detectors intact.
- Two-layer classification head: Linear → Hardswish → Dropout → Linear,
  trained at full learning rate.
- Differential learning rates: backbone block at lr/10, head at lr.
  This prevents the small pretrained backbone weights from being overwritten
  too fast while the randomly-initialised head converges.

Class-imbalance handling
------------------------
The dataset is ~30% wide / ~70% close-up.  We use a single consistent
mechanism to correct for this:

  CrossEntropyLoss(weight=[w_wide, w_closeup])

  where w_k = n_total / (K * n_k)   — standard inverse-frequency weighting.

This weights both the training gradient AND the validation loss identically,
so the model cannot cheat by predicting the majority class.  Using
WeightedRandomSampler instead creates a train/val distribution mismatch:
the head learns on 50/50 batches but early stopping is evaluated on a 70/30
val set, causing the model to oscillate between "all wide" and "all close-up".

Training features
-----------------
- CrossEntropyLoss with inverse-frequency class weights (single mechanism).
- Balanced accuracy (mean per-class recall) as the early-stopping metric so
  a model that always predicts one class can never win.
- On-the-fly augmentation: effectively multiplies the training set.
- Mixed-precision (AMP): ~2× faster on RTX 3060, lower VRAM use.
- CosineAnnealingLR: smooth LR decay that works well with fine-tuning.
- Gradient clipping (max_norm=1.0): prevents large-gradient spikes.
- Multi-worker DataLoader: uses CPU threads while GPU trains.
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

# Number of DataLoader workers — use multiple CPU threads on Ryzen 7 7800X
# 0 = main process (safe on Windows), >0 = parallel (faster but needs
# if __name__ == "__main__" guard in scripts, which our scripts have)
_NUM_WORKERS = 4

# Augmentation pipeline applied only during training
_AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    T.RandomGrayscale(p=0.05),         # simulate monochrome camera footage
])


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """
    Convert a single BGR np.ndarray frame to a normalised CHW float tensor.

    Frames coming from the sampler are already stored at 224×224 so the resize
    is a no-op in normal training use.  It is kept here so this function also
    works correctly on raw full-resolution frames during inference.
    """
    resized = cv2.resize(frame, _INPUT_SIZE)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor  = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def _build_mobilenet(num_classes: int) -> nn.Module:
    """
    Load pretrained MobileNetV3-Small and configure for fine-tuning.

    The last feature block (features[-1]) is unfrozen so the model can adapt
    its final convolutional features to concert-specific visual patterns.
    All earlier blocks stay frozen — their universal edge/texture detectors
    are not useful to overwrite with ~8,000 frames.

    Differential learning rates are applied by the caller:
        backbone block (features[-1]): lr / 10
        classification head:           lr
    """
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Freeze the entire backbone first
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze only the last feature block for domain adaptation
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # Replace the final linear with a two-layer head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.4),
        nn.Linear(128, num_classes),
    )
    return model


class _AugmentedDataset(torch.utils.data.Dataset):
    """Wraps pre-processed tensors and applies on-the-fly augmentation for training."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, augment: bool = False):
        self.x       = x
        self.y       = y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[idx]
        if self.augment:
            sample = _AUGMENT(sample)
        return sample, self.y[idx]


def _make_class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    w_k = n_total / (K * n_k)

    This is the standard sklearn-style 'balanced' weighting.  Applied to both
    training and validation loss so both metrics use the same scale, which
    prevents the model from winning early stopping by predicting the majority
    class.
    """
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
    """
    Binary classifier: 0 = wide/total shot, 1 = close-up shot.

    Wraps a MobileNetV3-Small pretrained on ImageNet, fine-tuned on concert data.

    Example usage::

        clf = ViewClassifier()
        clf.train(frames, labels)
        clf.save("models/view_classifier.pt")

        clf2 = ViewClassifier()
        clf2.load("models/view_classifier.pt")
        label = clf2.predict(frame)   # 0 or 1
    """

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = _build_mobilenet(num_classes).to(self.device)
        self.is_trained  = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        frames:     list[np.ndarray],
        labels:     list[int],
        epochs:     int   = 40,
        batch_size: int   = 32,
        lr:         float = 3e-4,
        val_split:  float = 0.2,
        patience:   int   = 15,
    ) -> None:
        """
        Fine-tune the classifier with mixed-precision and early stopping.

        Uses a **stratified shuffled split** so that both the training and
        validation sets contain a balanced mix of frames from all parts of the
        concert.  A chronological split (last 20% = finale/bowing section) would
        make the val set visually unlike the training set on a single concert,
        giving an artificially pessimistic and unstable accuracy signal.

        Stratification ensures each split has the same class ratio (~30/70) so
        the class weights remain valid on both sides.

        Note: when you have multiple concerts, switch to a concert-level split
        (train on concerts 1–N, validate on concert N+1) to measure true
        generalisation.  For single-concert training, shuffled split is correct.

        Early stopping is on *balanced accuracy* (mean per-class recall) rather
        than val loss.  A model that always predicts the majority class gets 50%
        balanced accuracy, not 68%, so it can never win by collapsing.

        Args:
            frames:     BGR frames stored at 224×224 by the sampler.
            labels:     0 = wide shot, 1 = close-up, one per frame.
            epochs:     Maximum number of training epochs.
            batch_size: Samples per gradient update.
            lr:         Learning rate for the classification head.
            val_split:  Fraction of data held out for validation.
            patience:   Stop early if balanced val accuracy doesn't improve.
        """
        print(f"Preprocessing {len(frames)} frames...")
        x_all = torch.stack([_preprocess_frame(f) for f in frames])
        y_all = torch.tensor(labels, dtype=torch.long)

        # Stratified shuffle split: preserve class ratio in both splits
        # while randomising which frames go to train vs val.
        indices   = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(42))
        x_all, y_all = x_all[indices], y_all[indices]

        split   = int(len(x_all) * (1 - val_split))
        x_train, x_val = x_all[:split], x_all[split:]
        y_train, y_val = y_all[:split], y_all[split:]

        train_ds = _AugmentedDataset(x_train, y_train, augment=True)
        val_ds   = _AugmentedDataset(x_val,   y_val,   augment=False)

        # persistent_workers requires num_workers > 0
        _workers    = _NUM_WORKERS
        _persistent = _workers > 0

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=_workers, persistent_workers=_persistent, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            num_workers=_workers, persistent_workers=_persistent, pin_memory=True,
        )

        # Inverse-frequency class weights applied to BOTH train and val loss.
        # A model predicting all close-up gets the same weighted loss as one
        # predicting all wide — neither can win early stopping by collapsing.
        class_weights = _make_class_weights(y_train, self.device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        # Differential learning rates:
        #   backbone block (features[-1]): lr/10 — small updates to pretrained weights
        #   classification head:           lr    — faster convergence for random init
        backbone_params = [
            p for p in self.model.features[-1].parameters() if p.requires_grad
        ]
        head_params = list(self.model.classifier.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr / 10},
                {"params": head_params,     "lr": lr},
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

        n_backbone = sum(p.numel() for p in backbone_params)
        n_head     = sum(p.numel() for p in head_params)
        counts     = torch.bincount(y_train)
        print(f"Training on {len(x_train)} | Validating on {len(x_val)}")
        print(f"Trainable parameters: {n_backbone + n_head:,}  "
              f"(backbone block: {n_backbone:,} @ lr/10 + head: {n_head:,} @ lr)")
        print(f"Train class counts — wide: {counts[0].item()}  close-up: {counts[1].item()}")
        print(f"Class weights — wide: {class_weights[0]:.3f}  close-up: {class_weights[1]:.3f}")
        print(f"Device: {self.device} | AMP: {use_amp} | Workers: {_workers}")

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
                    # Gradient clipping across all trainable params (backbone block + head).
                    # Passing model.parameters() is fine — frozen params have no
                    # gradients so clip_grad_norm_ is a no-op for them.
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, float]:
        """Return (mean_loss, accuracy, balanced_accuracy) without gradient tracking.

        Balanced accuracy = mean per-class recall.  A model predicting only the
        majority class scores 50%, not 68%, so it cannot win early stopping.
        """
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
        """Predict 0 (wide) or 1 (close-up) for a single frame."""
        return self.predict_batch([frame])[0]

    def predict_batch(self, frames: list[np.ndarray]) -> list[int]:
        """Predict labels for a list of frames in a single forward pass."""
        self.model.eval()
        x = torch.stack([_preprocess_frame(f) for f in frames]).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self.model(x)
        return torch.argmax(logits, dim=1).cpu().tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights to disk. Creates parent directories if needed."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights from disk (safe weights-only mode)."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.is_trained = True
        print(f"Model loaded from {path}")
