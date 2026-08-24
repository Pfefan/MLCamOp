"""Evaluate the model on held-out data."""

import argparse

import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data.sampler import generate_training_data, concert_split, split_dataset
from src.models.view_classifier import ViewClassifier
from src.utils.logger import setup_logger
from src.utils.metrics import cut_event_prf


def evaluate(config: dict) -> None:
    """Run the validation split and print metrics."""
    logger = setup_logger(config["logging"]["log_file"], config["logging"].get("level", "INFO"))

    concerts   = config["data"]["concerts"]
    sample_fps = config["data"].get("sample_fps", 1.0)
    cache_dir  = config["data"].get("cache_dir", "data/cache")
    dual_frame = config["data"].get("dual_frame", True)
    labeling   = config["data"].get("labeling", {})
    audio_cfg  = config["data"].get("audio", {})
    val_split  = config["training"].get("val_split", 0.2)

    logger.info("Loading frames (from cache if available)...")

    concert_sets: list = []
    for concert in concerts:
        threshold = concert.get("similarity_threshold",
                                config["data"].get("similarity_threshold", 39))
        concert_sets.append(generate_training_data(
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
        ))

    if not any(len(s) > 0 for s in concert_sets):
        logger.error("No frames loaded. Check video paths.")
        return

    if len(concerts) >= 2:
        _, val_data = concert_split(concert_sets, val_concert_idx=-1)
        ordered = True
        logger.info(
            "Evaluating on concert %d (%d frames) — held out during training.",
            len(concerts), len(val_data),
        )
    else:
        _, val_data = split_dataset(concert_sets[0], val_split=val_split)
        ordered = False
        logger.info("Evaluating on %d held-out frames...", len(val_data))

    val_frames, val_labels = val_data.frames, val_data.labels

    logger.info("Loading model from %s...", config["model"]["checkpoint"])
    classifier = ViewClassifier()
    classifier.load(config["model"]["checkpoint"])

    predictions = classifier.predict_batch(val_frames, audio=val_data.audio)

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

    if ordered:
        sample_fps    = config["data"].get("sample_fps", 1.0)
        tolerance_sec = config["evaluation"].get("cut_tolerance_sec", 0.4)
        tolerance     = max(1, round(tolerance_sec * sample_fps))
        ev = cut_event_prf(list(val_labels), list(predictions), tolerance)
        print(f"\nCut events (±{tolerance_sec:.2f}s = ±{tolerance} samples):")
        print(f"  Ground-truth cuts: {ev['n_gt']}  |  Predicted cuts: {ev['n_pred']}  |  Matched: {ev['tp']}")
        print(f"  Precision: {ev['precision'] * 100:.1f}%  Recall: {ev['recall'] * 100:.1f}%  F1: {ev['f1'] * 100:.1f}%")
        logger.info("Cut-event F1: %.1f%%", ev["f1"] * 100)
    else:
        print("\n(Cut-event metrics skipped — single-concert split is shuffled, not temporal.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained ViewClassifier.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)
