"""Train the view classifier."""

import argparse
import gc
import os

import torch
import yaml

from src.data.sampler import (
    generate_training_data, split_dataset, concat_samplesets,
)
from src.data.audio import frame_audio_feature_dim
from src.models.view_classifier import ViewClassifier, _NUM_WORKERS
from src.utils.logger import setup_logger
from src.utils.metrics import balanced_accuracy, cut_event_prf


def train_model(config: dict) -> None:
    """Load data, train, and save the checkpoint."""
    logger = setup_logger(config["logging"]["log_file"], config["logging"].get("level", "INFO"))

    concerts      = config["data"]["concerts"]
    sample_fps    = config["data"].get("sample_fps", 1.0)
    cache_dir     = config["data"].get("cache_dir", "data/cache")
    force_rebuild = config["data"].get("force_rebuild", False)
    dual_frame    = config["data"].get("dual_frame", True)
    labeling      = config["data"].get("labeling", {})
    audio_cfg     = config["data"].get("audio", {})
    val_split     = config["training"].get("val_split", 0.2)
    num_workers   = config["training"].get("num_workers", _NUM_WORKERS)
    batch_size    = config["training"]["batch_size"]

    concert_sets: list = []

    for i, concert in enumerate(concerts, start=1):
        threshold = concert.get("similarity_threshold",
                                config["data"].get("similarity_threshold", 39))
        logger.info("Concert %d/%d — loading frames (threshold=%s)...", i, len(concerts), threshold)
        logger.info("  total_view : %s", concert["total_view"])
        logger.info("  closeup    : %s", concert["closeup"])
        logger.info("  result     : %s", concert["result"])

        concert_sets.append(generate_training_data(
            total_view_path      = concert["total_view"],
            closeup_path         = concert["closeup"],
            result_path          = concert["result"],
            sample_fps           = sample_fps,
            similarity_threshold = threshold,
            cache_dir            = cache_dir,
            force_rebuild        = force_rebuild,
            dual_frame           = dual_frame,
            keep_transitions     = labeling.get("keep_transitions", True),
            transition_weight    = labeling.get("transition_weight", 0.3),
            audio_enabled        = audio_cfg.get("enabled", False),
            audio_sample_rate    = audio_cfg.get("sample_rate", 16000),
            audio_n_mels         = audio_cfg.get("n_mels", 32),
            cache_id             = concert.get("id"),
        ))

    total_frames = sum(len(s) for s in concert_sets)
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
        train_concerts = concert_sets[:-1]
        val_data       = concert_sets[-1]
    else:
        if force_random:
            logger.info("Using random frame-level split (--random-split diagnostic mode).")
        else:
            logger.info("Single concert: using shuffled 80/20 frame-level split.")
        train_ss, val_data = split_dataset(
            concat_samplesets(concert_sets), val_split=val_split
        )
        train_concerts = [train_ss]

    del concert_sets
    gc.collect()

    n_train = sum(len(c) for c in train_concerts)
    logger.info("Split: %d train frames / %d val frames", n_train, len(val_data))

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
    base, ext    = os.path.splitext(config["model"]["checkpoint"])
    cutf1_ckpt   = f"{base}_cutf1{ext}"

    classifier.train(
        train_concerts = train_concerts,
        val_data       = val_data,
        epochs         = config["training"]["epochs"],
        batch_size     = batch_size,
        lr             = config["training"]["learning_rate"],
        patience       = config["training"].get("patience", 15),
        num_workers    = num_workers,
        use_compile    = config["training"].get("compile", True),
        cut_tolerance_samples = tolerance,
        smooth_penalty        = smooth_pen,
        cutf1_checkpoint      = cutf1_ckpt,
    )

    save_path = config["model"]["checkpoint"]
    classifier.save(save_path)
    logger.info("Model saved to %s", save_path)

    _report_per_concert(config, classifier, logger)
    logger.info("Training complete.")


def _report_per_concert(config: dict, classifier: ViewClassifier, logger) -> None:
    """Score each concert independently for a quick overfit check."""
    concerts      = config["data"]["concerts"]
    sample_fps    = config["data"].get("sample_fps", 1.0)
    cache_dir     = config["data"].get("cache_dir", "data/cache")
    dual_frame    = config["data"].get("dual_frame", True)
    labeling      = config["data"].get("labeling", {})
    audio_cfg     = config["data"].get("audio", {})
    tolerance     = max(1, round(config["evaluation"].get("cut_tolerance_sec", 0.4) * sample_fps))

    logger.info("Per-concert validation (each concert scored independently):")
    for i, concert in enumerate(concerts, start=1):
        threshold = concert.get("similarity_threshold",
                                config["data"].get("similarity_threshold", 39))
        data = generate_training_data(
            total_view_path      = concert["total_view"],
            closeup_path         = concert["closeup"],
            result_path          = concert["result"],
            sample_fps           = sample_fps,
            similarity_threshold = threshold,
            cache_dir            = cache_dir,
            dual_frame           = dual_frame,
            keep_transitions     = labeling.get("keep_transitions", True),
            transition_weight    = labeling.get("transition_weight", 0.3),
            audio_enabled        = audio_cfg.get("enabled", False),
            audio_sample_rate    = audio_cfg.get("sample_rate", 16000),
            audio_n_mels         = audio_cfg.get("n_mels", 32),
            cache_id             = concert.get("id"),
        )
        preds = classifier.predict_batch(data.frames, audio=data.audio)
        bal   = balanced_accuracy(data.labels, preds)
        ev    = cut_event_prf(data.labels, preds, tolerance)
        logger.info(
            "  Concert %d: balAcc %.1f%% | cut F1 %.1f%% (gt %d / pred %d)",
            i, bal * 100, ev["f1"] * 100, ev["n_gt"], ev["n_pred"],
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

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    train_model(cfg)

