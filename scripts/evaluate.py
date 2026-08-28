"""Evaluate the model on held-out data."""

import argparse

import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data.sampler import generate_training_data, split_dataset
from src.models.view_classifier import ViewClassifier
from src.utils.logger import setup_logger
from src.utils.metrics import (
    cut_event_prf,
    pacing_stats,
    flicker_count,
    cut_onset_alignment,
    cut_indices,
    viterbi_smooth,
    calibrate_switch_penalty,
)


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

    def _load_concert(concert: dict):
        threshold = concert.get("similarity_threshold",
                                config["data"].get("similarity_threshold", 39))
        return generate_training_data(
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

    # Evaluation only needs the held-out concert; skip loading train concerts to
    # avoid holding every frame of every concert in RAM at once.
    if len(concerts) >= 2:
        held_out = concerts[-1]
        val_data = _load_concert(held_out)
        ordered  = True
        logger.info(
            "Evaluating on held-out concert '%s' (%d frames) — train concerts not loaded.",
            held_out.get("id", "last"), len(val_data),
        )
    else:
        full = _load_concert(concerts[0])
        _, val_data = split_dataset(full, val_split=val_split)
        del full
        ordered = False
        logger.info("Evaluating on %d held-out frames...", len(val_data))

    if len(val_data) == 0:
        logger.error("No frames loaded. Check video paths.")
        return

    val_frames, val_labels = val_data.frames, val_data.labels

    logger.info("Loading model from %s...", config["model"]["checkpoint"])
    classifier = ViewClassifier()
    classifier.load(config["model"]["checkpoint"])

    probs       = classifier.predict_proba_batch(val_frames, audio=val_data.audio)
    predictions = probs.argmax(axis=1).astype(int).tolist()

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
        min_shot      = max(1, int(0.5 * sample_fps))
        gt_pace       = pacing_stats(val_labels, sample_fps)
        gt_align      = (
            cut_onset_alignment(val_labels, val_data.audio, sample_fps)
            if val_data.audio is not None else None
        )

        def report_preds(title: str, preds: list[int]) -> None:
            """Print cut-event, pacing, jitter and onset metrics for a prediction stream."""
            ev = cut_event_prf(val_labels, preds, tolerance)
            print(f"\n{title}")
            print(f"Cut events (±{tolerance_sec:.2f}s = ±{tolerance} samples):")
            print(f"  Ground-truth cuts: {ev['n_gt']}  |  Predicted cuts: {ev['n_pred']}  |  Matched: {ev['tp']}")
            print(f"  Precision: {ev['precision'] * 100:.1f}%  Recall: {ev['recall'] * 100:.1f}%  F1: {ev['f1'] * 100:.1f}%")
            logger.info("%s cut-event F1: %.1f%%", title, ev["f1"] * 100)

            pr_pace = pacing_stats(preds, sample_fps)
            print(f"Pacing — GT vs Pred:")
            print(f"                      {'GT':>10}   {'Pred':>10}")
            print(f"  Cuts / minute:      {gt_pace['cuts_per_min']:>10.2f}   {pr_pace['cuts_per_min']:>10.2f}")
            print(f"  Shots:              {gt_pace['n_shots']:>10d}   {pr_pace['n_shots']:>10d}")
            print(f"  Median shot (s):    {gt_pace['median_shot_s']:>10.2f}   {pr_pace['median_shot_s']:>10.2f}")
            print(f"  P10 shot (s):       {gt_pace['p10_shot_s']:>10.2f}   {pr_pace['p10_shot_s']:>10.2f}")
            print(f"  P90 shot (s):       {gt_pace['p90_shot_s']:>10.2f}   {pr_pace['p90_shot_s']:>10.2f}")

            print(f"Jitter (shots shorter than {min_shot} samples / {min_shot / sample_fps:.2f}s):")
            print(f"  GT flickers: {flicker_count(val_labels, min_shot)}  |  Pred flickers: {flicker_count(preds, min_shot)}")

            if val_data.audio is not None and gt_align is not None:
                pr_align = cut_onset_alignment(preds, val_data.audio, sample_fps)
                print(f"Musical alignment (cut → nearest audio onset):")
                for name, al in (("GT", gt_align), ("Pred", pr_align)):
                    if al["median_dist_s"] is None:
                        print(f"  {name}: no onsets/cuts to align")
                    else:
                        print(f"  {name}: median dist {al['median_dist_s']:.2f}s  |  on-beat {al['frac_on_beat'] * 100:.1f}%")

        report_preds("RAW (per-frame argmax)", predictions)

        # Sweep the switch penalty so the effect on pacing/cut-F1 is visible.
        smooth_cfg   = config.get("evaluation", {}).get("smoothing", {})
        sweep        = smooth_cfg.get("penalty_sweep", [0.5, 1.0, 2.0, 4.0, 8.0])
        print(f"\n{'-' * 60}")
        print("SMOOTHING PENALTY SWEEP (Viterbi over class probabilities)")
        print(f"  {'penalty':>8}  {'cuts/min':>9}  {'cuts':>6}  {'cut-F1':>7}")
        print(f"  {'GT':>8}  {gt_pace['cuts_per_min']:>9.2f}  {gt_pace['n_cuts']:>6d}  {'—':>7}")
        for pen in sweep:
            sm  = viterbi_smooth(probs, float(pen))
            ev  = cut_event_prf(val_labels, sm, tolerance)
            pc  = pacing_stats(sm, sample_fps)
            print(f"  {pen:>8.2f}  {pc['cuts_per_min']:>9.2f}  {pc['n_cuts']:>6d}  {ev['f1'] * 100:>6.1f}%")

        # Final smoothed stream: fixed penalty, else auto-calibrated to GT pacing.
        fixed_pen = smooth_cfg.get("cut_penalty")
        if fixed_pen is not None:
            penalty = float(fixed_pen)
            how     = f"config cut_penalty={penalty:.2f}"
        else:
            penalty = calibrate_switch_penalty(probs, target_cuts=len(cut_indices(val_labels)))
            how     = f"auto-calibrated to GT pacing (penalty={penalty:.2f})"

        smoothed = viterbi_smooth(probs, penalty)
        print(f"\n{'-' * 60}")
        report_preds(f"SMOOTHED — {how}", smoothed)
    else:
        print("\n(Cut-event metrics skipped — single-concert split is shuffled, not temporal.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained ViewClassifier.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)
