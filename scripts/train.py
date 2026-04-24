"""Train the ViewClassifier on concert video data."""

import argparse
import gc

import numpy as np
import torch
import yaml

from src.data.sampler import generate_training_data, concert_split, split_dataset
from src.models.view_classifier import ViewClassifier, _NUM_WORKERS
from src.utils.logger import setup_logger


def train_model(config: dict) -> None:
    """Load concert data, split train/val, train ViewClassifier, save checkpoint."""
    logger = setup_logger(config["logging"]["log_file"])

    concerts      = config["data"]["concerts"]
    sample_fps    = config["data"].get("sample_fps", 1.0)
    cache_dir     = config["data"].get("cache_dir", "data/cache")
    force_rebuild = config["data"].get("force_rebuild", False)
    val_split     = config["training"].get("val_split", 0.2)
    low_memory    = config["training"].get("low_memory", False)
    num_workers   = 0 if low_memory else config["training"].get("num_workers", _NUM_WORKERS)
    batch_size    = config["training"]["batch_size"]
    if low_memory:
        batch_size = max(16, batch_size // 2)
        logger.info("low_memory mode: workers=0, batch_size=%d", batch_size)

    # Collect frames per concert (kept separate for concert-level splitting)
    concerts_frames: list[list] = []
    concerts_labels: list[list] = []

    for i, concert in enumerate(concerts, start=1):
        threshold = concert.get("similarity_threshold",
                                config["data"].get("similarity_threshold", 39))
        logger.info("Concert %d/%d — loading frames (threshold=%s)...", i, len(concerts), threshold)
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
            dual_frame           = True,
        )
        concerts_frames.append(frames)
        concerts_labels.append(labels)

    total_frames = sum(len(f) for f in concerts_frames)
    if total_frames == 0:
        logger.error("No frames sampled. Check video paths in configs/model_config.yaml.")
        return

    logger.info("Total dataset: %d frames from %d concert(s).", total_frames, len(concerts))

    force_random = config.get("_force_random_split", False)

    if len(concerts) >= 2 and not force_random:
        logger.info(
            "Using concert-level split: training on concerts 1–%d, validating on concert %d.",
            len(concerts) - 1, len(concerts),
        )
        train_frames, train_labels, val_frames, val_labels = concert_split(
            concerts_frames, concerts_labels, val_concert_idx=-1
        )
    else:
        if force_random:
            logger.info("Using random frame-level split (--random-split diagnostic mode).")
        else:
            logger.info("Single concert: using shuffled 80/20 frame-level split.")
        all_frames = np.concatenate(concerts_frames)
        all_labels = [l for cl in concerts_labels for l in cl]
        train_frames, train_labels, val_frames, val_labels = split_dataset(
            all_frames, all_labels, val_split=val_split
        )

    # Free per-concert lists — train/val now hold the only references we need.
    del concerts_frames, concerts_labels
    gc.collect()

    logger.info(
        "Split: %d train frames / %d val frames",
        len(train_frames), len(val_frames),
    )

    classifier = ViewClassifier()
    logger.info("Using device: %s", classifier.device)

    classifier.train(
        train_frames = train_frames,
        train_labels = train_labels,
        val_frames   = val_frames,
        val_labels   = val_labels,
        epochs       = config["training"]["epochs"],
        batch_size   = batch_size,
        lr           = config["training"]["learning_rate"],
        patience     = config["training"].get("patience", 15),
        num_workers  = num_workers,
    )

    save_path = config["model"]["checkpoint"]
    classifier.save(save_path)
    logger.info("Model saved to %s", save_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ViewClassifier.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument(
        "--random-split", action="store_true",
        help="Use random 80/20 frame-level split instead of concert-level. "
             "Diagnostic only — tests if the task is learnable at all.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.random_split:
        cfg["_force_random_split"] = True

    # Allow TF32 on Ampere+ GPUs — ~2× faster matmuls with negligible precision loss
    torch.set_float32_matmul_precision("high")
    # cuDNN auto-tuner: finds fastest conv algorithm for the fixed input size.
    # ~10s overhead on first batch, then faster for all subsequent batches.
    torch.backends.cudnn.benchmark = True

    train_model(cfg)

