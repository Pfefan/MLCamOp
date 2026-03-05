"""Pipeline stage: run model inference over video data."""

from typing import Any, Optional


def run_inference(video_data: Any, model: Any) -> list:
    """Run a trained model over video data and return predictions."""
    if model is None:
        return []
    return model.predict(video_data)


def load_video_for_inference(video_path: str) -> Optional[Any]:
    """Load a video from disk for inference (placeholder)."""
    _ = video_path
    return None


def process_video(video_path: str, model: Any) -> list:
    """Load a video and run inference on it end-to-end."""
    video_data = load_video_for_inference(video_path)
    if video_data is None:
        return []
    return run_inference(video_data, model)