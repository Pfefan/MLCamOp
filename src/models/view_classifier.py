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

# DataLoader workers.  With on-the-fly preprocessing in __getitem__,
# workers overlap CPU preprocessing with GPU training.  Each worker gets a
# fork of the process; numpy arrays are copy-on-write so memory is shared.
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
    """Convert a BGR uint8 frame to a normalised CHW float tensor."""
    resized = cv2.resize(frame, _INPUT_SIZE)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor  = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def _build_mobilenet(num_classes: int) -> nn.Module:
    """Load pretrained MobileNetV3-Small, freeze all but last two feature blocks."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Freeze the entire backbone first
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the last two feature blocks for domain adaptation
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.features[-2].parameters():
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


class _FrameDataset(torch.utils.data.Dataset):
    """Dataset that stores raw numpy frames and preprocesses on-the-fly.

    Keeping frames as uint8 numpy arrays (150 KB each) instead of float32
    tensors (600 KB each) cuts memory by 4x.  Preprocessing one frame per
    __getitem__ call is fast enough that DataLoader workers hide the cost.
    """

    def __init__(
        self,
        frames: list[np.ndarray],
        labels: list[int],
        augment: bool = False,
    ):
        self.frames  = frames
        self.labels  = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor = _preprocess_frame(self.frames[idx])
        if self.augment:
            tensor = _AUGMENT(tensor)
        return tensor, self.labels[idx]


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

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = _build_mobilenet(num_classes).to(self.device)
        self.is_trained  = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_frames: list[np.ndarray],
        train_labels: list[int],
        val_frames:   list[np.ndarray],
        val_labels:   list[int],
        epochs:       int   = 40,
        batch_size:   int   = 32,
        lr:           float = 3e-4,
        patience:     int   = 15,
    ) -> None:
        """Fine-tune with mixed-precision, early stopping on balanced accuracy."""
        train_ds = _FrameDataset(train_frames, train_labels, augment=True)
        val_ds   = _FrameDataset(val_frames,   val_labels,   augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=_NUM_WORKERS, pin_memory=True,
            persistent_workers=_NUM_WORKERS > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            num_workers=_NUM_WORKERS, pin_memory=True,
            persistent_workers=_NUM_WORKERS > 0,
        )

        y_train = torch.tensor(train_labels, dtype=torch.long)
        class_weights = _make_class_weights(y_train, self.device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        # Differential learning rates — three tiers:
        #   features[-2]: lr/100 — very slow updates to earlier pretrained block
        #   features[-1]: lr/10  — moderate updates to final conv block
        #   classifier head: lr  — full speed for randomly-initialised head
        backbone_params_deep   = [
            p for p in self.model.features[-2].parameters() if p.requires_grad
        ]
        backbone_params_top = [
            p for p in self.model.features[-1].parameters() if p.requires_grad
        ]
        head_params = list(self.model.classifier.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params_deep, "lr": lr / 100},
                {"params": backbone_params_top,  "lr": lr / 10},
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

        n_deep = sum(p.numel() for p in backbone_params_deep)
        n_top  = sum(p.numel() for p in backbone_params_top)
        n_head = sum(p.numel() for p in head_params)
        counts = torch.bincount(y_train)
        print(f"Training on {len(train_frames)} | Validating on {len(val_frames)}")
        print(f"Trainable parameters: {n_deep + n_top + n_head:,}  "
              f"(features[-2]: {n_deep:,} @ lr/100 | "
              f"features[-1]: {n_top:,} @ lr/10 | "
              f"head: {n_head:,} @ lr)")
        print(f"Train class counts — wide: {counts[0].item()}  close-up: {counts[1].item()}")
        print(f"Class weights — wide: {class_weights[0]:.3f}  close-up: {class_weights[1]:.3f}")
        print(f"Device: {self.device} | AMP: {use_amp} | Workers: {_NUM_WORKERS}")

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
        """Predict 0 (wide) or 1 (close-up) for a single frame."""
        return self.predict_batch([frame])[0]

    def predict_batch(self, frames: list[np.ndarray], batch_size: int = 256) -> list[int]:
        """Predict labels for a list of frames, processing in chunks."""
        self.model.eval()
        all_preds: list[int] = []

        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            x = torch.stack([_preprocess_frame(f) for f in chunk]).to(self.device)
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
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights from disk (safe weights-only mode)."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.is_trained = True
        print(f"Model loaded from {path}")
