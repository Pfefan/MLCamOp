"""Evaluate a trained ViewClassifier on held-out data and print a report."""

import argparse

import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data.sampler import generate_training_data, concert_split, split_dataset
from src.models.view_classifier import ViewClassifier
from src.utils.logger import setup_logger


def evaluate(config: dict) -> None:
    """Run the trained model on the held-out validation split and print a report."""
    logger = setup_logger(config["logging"]["log_file"])

    concerts   = config["data"]["concerts"]
    sample_fps = config["data"].get("sample_fps", 1.0)
    cache_dir  = config["data"].get("cache_dir", "data/cache")
    val_split  = config["training"].get("val_split", 0.2)

    logger.info("Loading frames (from cache if available)...")

    concerts_frames: list[list] = []
    concerts_labels: list[list] = []
    for concert in concerts:
        threshold = concert.get("similarity_threshold",
                                config["data"].get("similarity_threshold", 39))
        frames, labels = generate_training_data(
            total_view_path      = concert["total_view"],
            closeup_path         = concert["closeup"],
            result_path          = concert["result"],
            sample_fps           = sample_fps,
            similarity_threshold = threshold,
            cache_dir            = cache_dir,
            dual_frame           = True,
        )
        concerts_frames.append(frames)
        concerts_labels.append(labels)

    if not any(len(f) > 0 for f in concerts_frames):
        logger.error("No frames loaded. Check video paths.")
        return

    if len(concerts) >= 2:
        _, _, val_frames, val_labels = concert_split(
            concerts_frames, concerts_labels, val_concert_idx=-1
        )
        logger.info(
            "Evaluating on concert %d (%d frames) — held out during training.",
            len(concerts), len(val_frames),
        )
    else:
        all_frames = concerts_frames[0]
        all_labels = concerts_labels[0]
        _, _, val_frames, val_labels = split_dataset(
            all_frames, all_labels, val_split=val_split
        )
        logger.info("Evaluating on %d held-out frames...", len(val_frames))

    logger.info("Loading model from %s...", config["model"]["checkpoint"])
    classifier = ViewClassifier()
    classifier.load(config["model"]["checkpoint"])

    predictions = classifier.predict_batch(val_frames)

    labels_arr = np.array(val_labels)
    preds_arr  = np.array(predictions)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(classification_report(
        labels_arr,
        preds_arr,
        target_names=["wide (0)", "close-up (1)"],
        zero_division=0,
    ))

    cm = confusion_matrix(labels_arr, preds_arr)
    print("Confusion Matrix:")
    print("                    Predicted wide  Predicted close-up")
    print(f"  True wide:              {cm[0][0]:4d}              {cm[0][1]:4d}")
    print(f"  True close-up:          {cm[1][0]:4d}              {cm[1][1]:4d}")

    acc     = np.mean(preds_arr == labels_arr)
    w_rec   = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    c_rec   = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    bal_acc = (w_rec + c_rec) / 2
    print(f"\nAccuracy:          {acc     * 100:.1f}%")
    print(f"Balanced accuracy: {bal_acc  * 100:.1f}%")
    print(f"Wide recall:       {w_rec    * 100:.1f}%")
    print(f"Close-up recall:   {c_rec    * 100:.1f}%")
    logger.info("Balanced accuracy: %.1f%%", bal_acc * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained ViewClassifier.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)
