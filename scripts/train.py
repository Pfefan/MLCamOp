"""Train the ViewClassifier on concert video data."""

import argparse

import yaml

from src.data.sampler import generate_training_data
from src.models.view_classifier import ViewClassifier
from src.utils.logger import setup_logger


def train_model(config: dict) -> None:
    """
    Full training pipeline:
      1. Sample frames from total_view + result videos.
      2. Label each frame (wide=0, close-up=1) by comparing to the result.
      3. Train a MobileNetV3 classifier on those labelled frames.
      4. Save the best model weights to disk.

    Args:
        config: Parsed YAML config dict from configs/model_config.yaml.
    """
    logger = setup_logger(config["logging"]["log_file"])

    logger.info("Generating training data from videos...")
    logger.info("  total_view : %s", config["data"]["total_view"])
    logger.info("  result     : %s", config["data"]["result"])

    frames, labels = generate_training_data(
        total_view_path=config["data"]["total_view"],
        result_path=config["data"]["result"],
        sample_fps=config["data"].get("sample_fps", 1.0),
    )

    if not frames:
        logger.error("No frames were sampled. Check your video paths in configs/model_config.yaml.")
        return

    logger.info("Training ViewClassifier on %d frames...", len(frames))
    classifier = ViewClassifier()
    logger.info("Using device: %s", classifier.device)

    classifier.train(
        frames=frames,
        labels=labels,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["learning_rate"],
    )

    save_path = "models/view_classifier.pt"
    classifier.save(save_path)
    logger.info("Model saved to %s", save_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ViewClassifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config YAML file.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_model(cfg)
