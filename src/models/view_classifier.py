"""CNN-based classifier: decides per-frame if we should show wide or close-up shot."""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ViewClassifier:
    """
    Binary classifier: 0 = total/wide shot, 1 = close-up shot.
    Uses a lightweight CNN suitable for local training on a 3060.
    """

    def __init__(self, input_shape: tuple = (3, 224, 224), num_classes: int = 2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self) -> nn.Module:
        """Build a lightweight MobileNetV3-style CNN."""
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Replace final classifier for binary output
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        return model

    def train(
        self,
        frames: list[np.ndarray],
        labels: list[int],
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
    ) -> None:
        """
        Train the classifier on labelled frames.

        Args:
            frames: List of BGR frames as np.ndarray (H, W, 3).
            labels: List of int labels: 0=wide, 1=close-up.
            epochs: Number of training epochs.
            batch_size: Batch size.
            lr: Learning rate.
        """
        x = self._preprocess_frames(frames)
        y = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

        self.is_trained = True

    def predict(self, frame: np.ndarray) -> int:
        """
        Predict whether a frame should be wide (0) or close-up (1).

        Args:
            frame: Single BGR frame as np.ndarray.

        Returns:
            0 for wide shot, 1 for close-up.
        """
        self.model.eval()
        with torch.no_grad():
            x = self._preprocess_frames([frame]).to(self.device)
            outputs = self.model(x)
            return int(torch.argmax(outputs, dim=1).item())

    def predict_batch(self, frames: list[np.ndarray]) -> list[int]:
        """Predict labels for a list of frames."""
        self.model.eval()
        with torch.no_grad():
            x = self._preprocess_frames(frames).to(self.device)
            outputs = self.model(x)
            return torch.argmax(outputs, dim=1).cpu().tolist()

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.is_trained = True
        print(f"Model loaded from {path}")

    def _preprocess_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Convert BGR np.ndarray frames to normalised CHW float tensors."""
        tensors = []
        for frame in frames:
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            # ImageNet normalisation
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors)
