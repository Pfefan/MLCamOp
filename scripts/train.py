"""Train the ViewClassifier on concert video data."""

import argparse

import yaml

from src.data.sampler import generate_training_data
from src.models.view_classifier import ViewClassifier
from src.utils.logger import setup_logger


def train_model(config: dict) -> None:
    """
    Full training pipeline:
      1. Sample frames from each concert's total_view + closeup + result (cached).
      2. Concatenate all concerts into one dataset.
      3. Train with class-balanced loss, augmentation, AMP, early stopping.
      4. Save best checkpoint to disk.
    """
    logger = setup_logger(config["logging"]["log_file"])

    concerts   = config["data"]["concerts"]
    sample_fps = config["data"].get("sample_fps", 1.0)
    threshold  = config["data"].get("similarity_threshold", 65.9)
    cache_dir  = config["data"].get("cache_dir", "data/cache")
    force_rebuild = config["data"].get("force_rebuild", False)

    all_frames: list = []
    all_labels: list = []

    for i, concert in enumerate(concerts, start=1):
        logger.info("Concert %d/%d — loading frames...", i, len(concerts))
        logger.info("  total_view : %s", concert["total_view"])
        logger.info("  closeup    : %s", concert["closeup"])
        logger.info("  result     : %s", concert["result"])

        frames, labels = generate_training_data(
            total_view_path      = concert["total_view"],
            closeup_path         = concert["closeup"],
            result_path          = concert["result"],
            sample_fps           = sample_fps,
            similarity_threshold = threshold,
            cache_dir            = cache_dir,
            force_rebuild        = force_rebuild,
        )
        all_frames.extend(frames)
        all_labels.extend(labels)

    if not all_frames:
        logger.error("No frames sampled. Check video paths in configs/model_config.yaml.")
        return

    logger.info(
        "Total training data: %d frames from %d concert(s).",
        len(all_frames), len(concerts),
    )

    classifier = ViewClassifier()
    logger.info("Using device: %s", classifier.device)

    classifier.train(
        frames     = all_frames,
        labels     = all_labels,
        epochs     = config["training"]["epochs"],
        batch_size = config["training"]["batch_size"],
        lr         = config["training"]["learning_rate"],
        val_split  = config["training"].get("val_split", 0.2),
        patience   = config["training"].get("patience", 10),
    )

    save_path = config["model"]["checkpoint"]
    classifier.save(save_path)
    logger.info("Model saved to %s", save_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ViewClassifier.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_model(cfg)

