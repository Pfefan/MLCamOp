"""Train the view classifier."""

import argparse
import os
import random

import numpy as np
import torch
import yaml

from src.data.sampler import (
    SampleSet, load_concert, concert_name, val_concert_indices,
    split_dataset, concat_samplesets,
)
from src.data.audio import frame_audio_feature_dim
from src.models.view_classifier import ViewClassifier, _NUM_WORKERS
from src.utils.logger import setup_logger
from src.utils.metrics import balanced_accuracy, cut_event_prf


def train_model(config: dict) -> None:
    """Load data, train, and save the checkpoint."""
    logger = setup_logger(config["logging"]["log_file"], config["logging"].get("level", "INFO"))

    data_cfg      = config["data"]
    concerts      = data_cfg["concerts"]
    names         = [concert_name(concert, i) for i, concert in enumerate(concerts)]
    sample_fps    = data_cfg.get("sample_fps", 1.0)
    force_rebuild = data_cfg.get("force_rebuild", False)
    dual_frame    = data_cfg.get("dual_frame", True)
    audio_cfg     = data_cfg.get("audio", {})
    val_split     = config["training"].get("val_split", 0.2)
    num_workers   = config["training"].get("num_workers", _NUM_WORKERS)
    batch_size    = config["training"]["batch_size"]

    concert_sets: list[SampleSet] = []

    for i, concert in enumerate(concerts):
        logger.info("Concert %d/%d '%s' — loading frames...", i + 1, len(concerts), names[i])
        logger.info("  total_view : %s", concert["total_view"])
        logger.info("  closeup    : %s", concert["closeup"])
        logger.info("  result     : %s", concert["result"])
        concert_sets.append(load_concert(data_cfg, concert, force_rebuild=force_rebuild))

    total_frames = sum(len(s) for s in concert_sets)
    if total_frames == 0:
        logger.error("No frames sampled. Check video paths in configs/model_config.yaml.")
        return

    logger.info("Total dataset: %d frames from %d concert(s).", total_frames, len(concerts))

    force_random = config.get("_force_random_split", False)
    val_indices: list[int] = []

    if len(concerts) >= 2 and not force_random:
        val_indices    = val_concert_indices(data_cfg)
        train_concerts = [s for i, s in enumerate(concert_sets) if i not in val_indices]
        val_concerts   = [concert_sets[i] for i in val_indices]
        logger.info(
            "Concert-level split — train: %s | val: %s",
            ", ".join(name for i, name in enumerate(names) if i not in val_indices),
            ", ".join(names[i] for i in val_indices),
        )
    else:
        if force_random:
            logger.info("Using random frame-level split (--random-split diagnostic mode).")
        else:
            logger.info("Single concert: using shuffled 80/20 frame-level split.")
        train_ss, val_ss = split_dataset(
            concat_samplesets(concert_sets), val_split=val_split
        )
        train_concerts, val_concerts = [train_ss], [val_ss]

    n_train = sum(len(c) for c in train_concerts)
    n_val   = sum(len(c) for c in val_concerts)
    logger.info("Split: %d train frames / %d val frames", n_train, n_val)

    audio_dim = frame_audio_feature_dim(audio_cfg.get("n_mels", 32)) if audio_cfg.get("enabled", False) else 0
    classifier = ViewClassifier(
        num_classes = config["model"].get("num_classes", 2),
        dual_frame  = dual_frame,
        window      = config["temporal"].get("window", 5),
        lookahead   = config["temporal"].get("lookahead", 1),
        rnn_hidden  = config["temporal"].get("rnn_hidden", 128),
        audio_dim   = audio_dim,
        audio_sample_rate = audio_cfg.get("sample_rate", 16000),
    )
    logger.info("Using device: %s", classifier.device)

    eval_cfg     = config.get("evaluation", {})
    tolerance    = max(1, round(eval_cfg.get("cut_tolerance_sec", 0.4) * sample_fps))
    smooth_pen   = eval_cfg.get("smoothing", {}).get("cut_penalty")
    smooth_pen   = 0.1 if smooth_pen is None else float(smooth_pen)
    min_shot_sec = float(eval_cfg.get("smoothing", {}).get("min_shot_sec", 0.0) or 0.0)
    smooth_min   = max(1, round(min_shot_sec * sample_fps))
    base, ext    = os.path.splitext(config["model"]["checkpoint"])
    cutf1_ckpt   = f"{base}_cutf1{ext}"

    classifier.train(
        train_concerts = train_concerts,
        val_concerts   = val_concerts,
        epochs         = config["training"]["epochs"],
        batch_size     = batch_size,
        lr             = config["training"]["learning_rate"],
        patience       = config["training"].get("patience", 15),
        num_workers    = num_workers,
        use_compile    = config["training"].get("compile", True),
        cut_tolerance_samples = tolerance,
        smooth_penalty        = smooth_pen,
        cutf1_checkpoint      = cutf1_ckpt,
        label_smoothing       = config["training"].get("label_smoothing", 0.1),
        smooth_min_shot       = smooth_min,
        prefetch_factor       = config["training"].get("prefetch_factor", 2),
    )

    save_path = config["model"]["checkpoint"]
    classifier.save(save_path)
    logger.info("Model saved to %s", save_path)

    _report_per_concert(classifier, names, concert_sets, val_indices, tolerance, logger)
    logger.info("Training complete.")


def _report_per_concert(
    classifier:   ViewClassifier,
    names:        list[str],
    concert_sets: list[SampleSet],
    val_indices:  list[int],
    tolerance:    int,
    logger,
) -> None:
    """Score each concert independently; the train vs held-out gap shows overfitting."""
    logger.info("Per-concert scores (each concert scored independently):")
    for i, (name, data) in enumerate(zip(names, concert_sets)):
        preds = classifier.predict_batch(data.frames, audio=data.audio)
        bal   = balanced_accuracy(data.labels, preds)
        ev    = cut_event_prf(data.labels, preds, tolerance)
        logger.info(
            "  %-5s %s: balAcc %.1f%% | cut F1 %.1f%% (gt %d / pred %d)",
            "val" if i in val_indices else "train", name,
            bal * 100, ev["f1"] * 100, ev["n_gt"], ev["n_pred"],
        )


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

    seed = int(cfg["training"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    train_model(cfg)

