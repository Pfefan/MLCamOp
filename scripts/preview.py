"""Preview a cut using the trained model."""

import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.view_classifier import ViewClassifier
from src.pipeline.inference import classify_video
from src.pipeline.editing import _enforce_min_shot_length

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


def _simulate_closeup(frame: np.ndarray, crop_factor: float) -> np.ndarray:
    """Simulate a close-up by centre-cropping and scaling up."""
    h, w = frame.shape[:2]
    crop_h = int(h * crop_factor)
    crop_w = int(w * crop_factor)

    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2

    cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _render_preview(
    total_view_path: str,
    closeup_path:    str | None,
    output_path:     str,
    timestamps:      list[float],
    labels:          list[int],
    video_fps:       float,
    crop_factor:     float,
) -> None:
    cap_total = cv2.VideoCapture(total_view_path)
    if not cap_total.isOpened():
        raise ValueError(f"Cannot open video: {total_view_path}")

    cap_closeup = None
    if closeup_path:
        cap_closeup = cv2.VideoCapture(closeup_path)
        if not cap_closeup.isOpened():
            raise ValueError(f"Cannot open video: {closeup_path}")

    width  = int(cap_total.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_total.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_total.get(cv2.CAP_PROP_FRAME_COUNT))
    blank  = np.zeros((height, width, 3), dtype=np.uint8)
    video_only_path = f"{output_path}.video.mp4"

    writer = cv2.VideoWriter(
        video_only_path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        video_fps,
        (width, height),
    )

    sample_frames = [int(ts * video_fps) for ts in timestamps]
    sample_idx = 0
    current_label = labels[0] if labels else 0

    iterable = tqdm(range(total_frames), desc="  rendering", unit="frame") if _TQDM else range(total_frames)
    for frame_idx in iterable:
        while sample_idx + 1 < len(sample_frames) and frame_idx >= sample_frames[sample_idx + 1]:
            sample_idx += 1
            current_label = labels[sample_idx]

        if current_label == 1 and cap_closeup is not None:
            cap_total.grab()
            ok, frame = cap_closeup.read()
            frame = frame if ok else blank
        else:
            ok, total_frame = cap_total.read()
            total_frame = total_frame if ok else blank
            if cap_closeup is not None:
                cap_closeup.grab()
            frame = total_frame if current_label == 0 else _simulate_closeup(total_frame, crop_factor)
        writer.write(frame)

    writer.release()
    cap_total.release()
    if cap_closeup is not None:
        cap_closeup.release()

    _mux_audio(video_only_path, total_view_path, output_path)


def _mux_audio(video_only_path: str, audio_source_path: str, output_path: str) -> None:
    if shutil.which("ffmpeg") is None:
        os.replace(video_only_path, output_path)
        print("ffmpeg not found; wrote video without audio.")
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", video_only_path,
        "-i", audio_source_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(video_only_path)
    except subprocess.CalledProcessError:
        os.replace(video_only_path, output_path)
        print("Could not mux audio; wrote video without audio.")


def run_preview(
    total_view_path: str,
    output_path:     str,
    checkpoint:      str,
    sample_fps:      float,
    crop_factor:     float,
    closeup_path:    str | None = None,
    min_shot_sec:    float      = 1,
) -> None:
    """Classify shots and render a preview cut."""
    print(f"Loading model from {checkpoint} ...")
    classifier = ViewClassifier()
    classifier.load(checkpoint)

    if classifier.dual_frame and closeup_path is None:
        raise ValueError(
            "Checkpoint is a dual-frame model and needs a real close-up feed. "
            "Pass --closeup; the simulated digital zoom cannot be classified."
        )

    print(f"Classifying shots in {total_view_path} ...")
    predictions = classify_video(
        total_view_path, classifier, sample_fps=sample_fps, closeup_path=closeup_path,
    )

    timestamps = [ts    for ts, _     in predictions]
    labels     = [label for _,  label in predictions]

    n_wide    = labels.count(0)
    n_closeup = labels.count(1)
    print(
        f"Classified {len(labels)} sample points — "
        f"wide: {n_wide} ({n_wide / len(labels) * 100:.1f}%)  "
        f"close-up: {n_closeup} ({n_closeup / len(labels) * 100:.1f}%)"
    )

    cap = cv2.VideoCapture(total_view_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    min_shot_frames = max(1, int(min_shot_sec * sample_fps))
    smoothed_labels = _enforce_min_shot_length(labels, min_shot_frames)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Rendering full-rate preview to {output_path} (min shot = {min_shot_sec}s) ...")
    _render_preview(
        total_view_path, closeup_path, output_path, timestamps, smoothed_labels,
        video_fps, crop_factor,
    )
    print(f"Done.  Output written to: {output_path}")


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
