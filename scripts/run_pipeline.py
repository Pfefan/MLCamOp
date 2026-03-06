"""
End-to-end inference pipeline: classify shots in a new concert and render output.

Usage::

    # Run on the first concert defined in config (default)
    python311 scripts/run_pipeline.py --config configs/model_config.yaml \\
        --output output/result.mp4

    # Run on a specific concert (0-based index)
    python311 scripts/run_pipeline.py --config configs/model_config.yaml \\
        --concert 1 --output output/result_concert2.mp4

This script is the production entry point once the model is trained.
It does NOT re-train — run ``scripts/train.py`` for that.
For a quick visual sanity-check with a single video, use ``scripts/preview.py``.
"""

import argparse
import os

import cv2
import numpy as np
import yaml

from src.models.view_classifier import ViewClassifier
from src.pipeline.inference import classify_video
from src.pipeline.editing import assemble_cut
from src.postprocessing.renderer import render_video
from src.utils.logger import setup_logger


def _read_frames_at_timestamps(
    path:       str,
    timestamps: list[float],
    fps:        float,
) -> list[np.ndarray]:
    """
    Seek-based frame reader: extract frames at the given timestamps (seconds).

    Args:
        path:       Video file path.
        timestamps: Sample timestamps in seconds.
        fps:        Video FPS used to convert timestamps to frame indices.

    Returns:
        List of BGR frames, same length as *timestamps*.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    blank  = np.zeros((height, width, 3), dtype=np.uint8)

    frames: list[np.ndarray] = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        frames.append(frame if ret else blank)

    cap.release()
    return frames


def run_pipeline(config: dict, output_path: str, concert_index: int = 0) -> None:
    """
    Load the trained model, classify every sampled frame in total_view,
    assemble the cut from total_view + closeup, and write the result to disk.

    Args:
        config:        Loaded model_config.yaml dict.
        output_path:   Where to write the final MP4.
        concert_index: Which concert in ``config["data"]["concerts"]`` to process.
    """
    logger = setup_logger(config["logging"]["log_file"])

    concerts = config["data"]["concerts"]
    if concert_index >= len(concerts):
        raise IndexError(
            f"concert_index={concert_index} is out of range "
            f"({len(concerts)} concert(s) defined in config)."
        )
    concert = concerts[concert_index]
    logger.info(
        "Running pipeline on concert %d / %d", concert_index + 1, len(concerts)
    )

    total_view_path = concert["total_view"]
    closeup_path    = concert["closeup"]
    checkpoint      = config["model"]["checkpoint"]
    sample_fps      = config["data"].get("sample_fps", 1.0)

    logger.info("Loading model from %s", checkpoint)
    classifier = ViewClassifier()
    classifier.load(checkpoint)

    logger.info("Classifying shots in %s ...", total_view_path)
    predictions = classify_video(total_view_path, classifier, sample_fps=sample_fps)
    logger.info("Classified %d sample points", len(predictions))

    timestamps = [ts    for ts, _     in predictions]
    labels     = [label for _,  label in predictions]

    n_wide    = labels.count(0)
    n_closeup = labels.count(1)
    logger.info(
        "Label distribution — wide: %d (%.1f%%)  close-up: %d (%.1f%%)",
        n_wide,    n_wide    / len(labels) * 100,
        n_closeup, n_closeup / len(labels) * 100,
    )

    # Get source FPS for frame-index conversion
    cap = cv2.VideoCapture(total_view_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    logger.info("Reading total-view frames at sample timestamps ...")
    total_frames = _read_frames_at_timestamps(total_view_path, timestamps, video_fps)

    logger.info("Reading close-up frames at sample timestamps ...")
    closeup_frames = _read_frames_at_timestamps(closeup_path, timestamps, video_fps)

    # Minimum shot length: 0.5 s expressed in sample units
    min_shot_frames = max(1, int(0.5 * sample_fps))
    logger.info("Assembling cut (min shot = %d samples) ...", min_shot_frames)
    output_frames = assemble_cut(total_frames, closeup_frames, labels, min_shot_frames)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    logger.info("Rendering %d frames to %s ...", len(output_frames), output_path)
    render_video(output_frames, output_path, fps=sample_fps)
    logger.info("Pipeline complete.  Output written to: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full inference pipeline.")
    parser.add_argument("--config",  default="configs/model_config.yaml")
    parser.add_argument("--output",  default="output/result.mp4")
    parser.add_argument(
        "--concert",
        type=int,
        default=0,
        help="0-based index of the concert to process (default: 0).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_pipeline(cfg, args.output, concert_index=args.concert)
