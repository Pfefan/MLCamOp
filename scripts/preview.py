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
from src.pipeline.inference import classify_video_smoothed
from src.pipeline.editing import _enforce_min_shot_length

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


def _render_preview(
    total_view_path: str,
    closeup_path:    str | None,
    output_path:     str,
    timestamps:      list[float],
    labels:          list[int],
    video_fps:       float,
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
            ok, frame = cap_total.read()
            frame = frame if ok else blank
            if cap_closeup is not None:
                cap_closeup.grab()
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
    closeup_path:    str | None = None,
    min_shot_sec:    float      = 0.0,
    cut_penalty:     float      = 0.5,
) -> None:
    """Classify shots and render a preview cut."""
    print(f"Loading model from {checkpoint} ...")
    classifier = ViewClassifier()
    classifier.load(checkpoint)

    if classifier.dual_frame and closeup_path is None:
        raise ValueError(
            "Checkpoint is a dual-frame model and needs a real close-up feed. "
            "Pass --closeup or use --concert."
        )

    print(f"Classifying shots in {total_view_path} (Viterbi smoothing, penalty={cut_penalty}) ...")
    predictions = classify_video_smoothed(
        total_view_path, classifier, sample_fps=sample_fps,
        closeup_path=closeup_path, cut_penalty=cut_penalty,
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
    smoothed_labels = (
        _enforce_min_shot_length(labels, min_shot_frames) if min_shot_frames > 1 else labels
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Rendering full-rate preview to {output_path} ...")
    _render_preview(
        total_view_path, closeup_path, output_path, timestamps, smoothed_labels,
        video_fps,
    )
    print(f"Done.  Output written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render an auto-edited cut (full frame rate, with audio).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="configs/model_config.yaml",
        help="Path to model_config.yaml (default: configs/model_config.yaml)",
    )
    parser.add_argument(
        "--concert", type=int, default=None,
        help="0-based index of a concert in the config to render. "
             "Use this instead of --input/--closeup.",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to the wide-shot (total_view) video. "
             "Alternative to --concert.",
    )
    parser.add_argument(
        "--closeup", default=None,
        help="Path to the close-up video (required by dual-frame models). "
             "Alternative to --concert.",
    )
    parser.add_argument(
        "--output", default="output/preview.mp4",
        help="Where to write the output MP4 (default: output/preview.mp4)",
    )
    parser.add_argument(
        "--min-shot", type=float, default=0.0,
        dest="min_shot",
        help="Optional extra minimum cut duration in seconds on top of Viterbi "
             "smoothing (default: 0 = pure Viterbi, matches evaluation).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.concert is not None:
        concerts = cfg["data"]["concerts"]
        if not 0 <= args.concert < len(concerts):
            parser.error(f"--concert {args.concert} out of range (0..{len(concerts) - 1})")
        concert     = concerts[args.concert]
        total_view  = concert["total_view"]
        closeup     = concert["closeup"]
    elif args.input is not None:
        total_view  = args.input
        closeup     = args.closeup
    else:
        parser.error("provide either --concert N or --input (with --closeup)")

    _penalty = cfg.get("evaluation", {}).get("smoothing", {}).get("cut_penalty")
    _penalty = 0.5 if _penalty is None else float(_penalty)

    run_preview(
        total_view_path = total_view,
        output_path     = args.output,
        checkpoint      = cfg["model"]["checkpoint"],
        sample_fps      = cfg["data"].get("sample_fps", 1.0),
        closeup_path    = closeup,
        min_shot_sec    = args.min_shot,
        cut_penalty     = _penalty,
    )
