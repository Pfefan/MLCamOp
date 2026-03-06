"""Evaluate a trained ViewClassifier on held-out data and print a report."""

import argparse

import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data.sampler import generate_training_data
from src.models.view_classifier import ViewClassifier
from src.utils.logger import setup_logger


def evaluate(config: dict) -> None:
    """
    Load the trained model, run inference on a random 20% held-out split,
    and print a full classification report with confusion matrix.

    Uses the same cached frame data as training — no video re-read needed.
    Supports multiple concerts defined under data.concerts in the config.
    """
    logger = setup_logger(config["logging"]["log_file"])

    concerts      = config["data"]["concerts"]
    sample_fps    = config["data"].get("sample_fps", 1.0)
    threshold     = config["data"].get("similarity_threshold", 65.9)
    cache_dir     = config["data"].get("cache_dir", "data/cache")

    logger.info("Loading evaluation frames (from cache if available)...")

    all_frames: list = []
    all_labels: list = []
    for concert in concerts:
        frames, labels = generate_training_data(
            total_view_path      = concert["total_view"],
            closeup_path         = concert["closeup"],
            result_path          = concert["result"],
            sample_fps           = sample_fps,
            similarity_threshold = threshold,
            cache_dir            = cache_dir,
        )
        all_frames.extend(frames)
        all_labels.extend(labels)

    if not all_frames:
        logger.error("No frames loaded. Check video paths.")
        return

    # Use the last 20% as the test split (matches the shuffled train split
    # which uses a fixed seed=42, so these frames were not seen during training)
    split       = int(len(all_frames) * 0.8)
    test_frames = all_frames[split:]
    test_labels = all_labels[split:]

    logger.info("Loading model from %s...", config["model"]["checkpoint"])
    classifier = ViewClassifier()
    classifier.load(config["model"]["checkpoint"])

    logger.info("Running inference on %d test frames...", len(test_frames))
    predictions = classifier.predict_batch(test_frames)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(classification_report(
        test_labels,
        predictions,
        target_names=["wide (0)", "close-up (1)"],
        zero_division=0,
    ))

    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:")
    print("  Predicted wide | Predicted close-up")
    print(f"  True wide:      {cm[0][0]:4d}  |  {cm[0][1]:4d}")
    print(f"  True close-up:  {cm[1][0]:4d}  |  {cm[1][1]:4d}")

    acc = np.mean(np.array(predictions) == np.array(test_labels))
    logger.info("Test accuracy: %.1f%%", acc * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained ViewClassifier.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)
