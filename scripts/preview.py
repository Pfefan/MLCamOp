"""Preview: produce a cut video from a single wide-shot using the trained model.

A digital zoom simulates the close-up camera unless a real close-up is provided.

Usage::

    python scripts/preview.py --config configs/model_config.yaml \\
        --input "path/to/total_view.mp4" --output "output/preview.mp4"

    # With a real close-up camera:
    python scripts/preview.py --config configs/model_config.yaml \\
        --input "path/to/total_view.mp4" --closeup "path/to/closeup.mp4" \\
        --output "output/preview.mp4"
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

# Make sure the project root is on the path when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.view_classifier import ViewClassifier
from src.pipeline.inference import classify_video
from src.pipeline.editing import assemble_cut
from src.postprocessing.renderer import render_video

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
# Frame reading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_all_frames(path: str, desc: str) -> tuple[list[np.ndarray], float]:
    """Read every frame from *path* into memory and return (frames, fps)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[np.ndarray] = []
    iterable = tqdm(range(total_frames), desc=desc, unit="frame") if _TQDM else range(total_frames)

    for _ in iterable:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def _read_frames_at_timestamps(
    path:       str,
    timestamps: list[float],
    fps:        float,
    desc:       str,
) -> list[np.ndarray]:
    """Seek-based frame reader: extract frames at the given timestamps (seconds)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    frames: list[np.ndarray] = []
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    blank  = np.zeros((height, width, 3), dtype=np.uint8)

    iterable = tqdm(timestamps, desc=desc, unit="frame") if _TQDM else timestamps

    for ts in iterable:
        idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame if ret else blank)

    cap.release()
    return frames


def _simulate_closeup(frame: np.ndarray, crop_factor: float) -> np.ndarray:
    """Centre-crop then upscale to simulate a close-up (digital zoom)."""
    h, w = frame.shape[:2]
    crop_h = int(h * crop_factor)
    crop_w = int(w * crop_factor)

    # Centre the crop
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2

    cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_preview(
    total_view_path: str,
    output_path:     str,
    checkpoint:      str,
    sample_fps:      float,
    crop_factor:     float,
    closeup_path:    str | None = None,
    min_shot_sec:    float      = 0.5,
) -> None:
    """Classify, assemble, and render a preview cut from a wide-shot video."""
    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"Loading model from {checkpoint} ...")
    classifier = ViewClassifier()
    classifier.load(checkpoint)

    # ── 2. Classify the wide-shot video ──────────────────────────────────────
    print(f"Classifying shots in {total_view_path} ...")
    predictions = classify_video(total_view_path, classifier, sample_fps=sample_fps)

    timestamps = [ts    for ts, _     in predictions]
    labels     = [label for _,  label in predictions]

    n_wide    = labels.count(0)
    n_closeup = labels.count(1)
    print(
        f"Classified {len(labels)} sample points — "
        f"wide: {n_wide} ({n_wide / len(labels) * 100:.1f}%)  "
        f"close-up: {n_closeup} ({n_closeup / len(labels) * 100:.1f}%)"
    )

    # ── 3. Get the video FPS so we can resolve timestamps → frame indices ─────
    cap = cv2.VideoCapture(total_view_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # ── 4. Load source frames at the classified timestamps ───────────────────
    print("Reading total-view frames ...")
    total_frames = _read_frames_at_timestamps(
        total_view_path, timestamps, video_fps, desc="  total_view"
    )

    if closeup_path:
        print(f"Reading close-up frames from {closeup_path} ...")
        closeup_frames = _read_frames_at_timestamps(
            closeup_path, timestamps, video_fps, desc="  closeup"
        )
    else:
        crop_pct = int(crop_factor * 100)
        print(f"Simulating close-up with {crop_pct}% centre-crop (digital zoom) ...")
        iterable = (
            tqdm(total_frames, desc="  simulating crop", unit="frame")
            if _TQDM else total_frames
        )
        closeup_frames = [_simulate_closeup(f, crop_factor) for f in iterable]

    # ── 5. Assemble the cut ───────────────────────────────────────────────────
    # Convert min_shot_sec to frames using sample_fps (labels are at sample_fps)
    min_shot_frames = max(1, int(min_shot_sec * sample_fps))
    print(f"Assembling cut (min shot = {min_shot_sec}s = {min_shot_frames} samples) ...")
    output_frames = assemble_cut(total_frames, closeup_frames, labels, min_shot_frames)

    # ── 6. Render to disk ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Rendering {len(output_frames)} frames to {output_path} ...")

    # The assembled frames are at sample_fps (sparse).  Write them at sample_fps
    # so timing is correct — each output frame represents 1/sample_fps seconds.
    render_video(output_frames, output_path, fps=sample_fps)
    print(f"Done.  Output written to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preview the auto-edit on a single wide-shot video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="configs/model_config.yaml",
        help="Path to model_config.yaml (default: configs/model_config.yaml)",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the wide-shot (total_view) video.",
    )
    parser.add_argument(
        "--closeup", default=None,
        help="Optional: path to a real close-up video.  "
             "If omitted, a digital zoom is simulated from --input.",
    )
    parser.add_argument(
        "--output", default="output/preview.mp4",
        help="Where to write the output MP4 (default: output/preview.mp4)",
    )
    parser.add_argument(
        "--crop", type=float, default=0.5,
        help="Centre-crop fraction for the simulated close-up (default: 0.5 = 2× zoom).  "
             "Ignored if --closeup is provided.",
    )
    parser.add_argument(
        "--min-shot", type=float, default=0.5,
        dest="min_shot",
        help="Minimum cut duration in seconds (default: 0.5).  "
             "Shorter predicted shots are merged with the surrounding shot.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_preview(
        total_view_path = args.input,
        output_path     = args.output,
        checkpoint      = cfg["model"]["checkpoint"],
        sample_fps      = cfg["data"].get("sample_fps", 1.0),
        crop_factor     = args.crop,
        closeup_path    = args.closeup,
        min_shot_sec    = args.min_shot,
    )
