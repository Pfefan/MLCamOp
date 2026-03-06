"""Pipeline stage: ingest raw video files and validate they can be opened."""

import cv2


def load_video(path: str) -> cv2.VideoCapture:
    """
    Open a video file and return a ``VideoCapture`` handle.

    Args:
        path: File path to the video.

    Returns:
        Opened ``cv2.VideoCapture`` instance.  Caller is responsible for
        calling ``cap.release()`` when done, or using it as a context.

    Raises:
        ValueError: If the file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    return cap


def ingest_videos(video_paths: list[str]) -> list[cv2.VideoCapture]:
    """
    Open multiple video files and return their ``VideoCapture`` handles.

    Args:
        video_paths: Paths to the video files to ingest.

    Returns:
        List of opened ``VideoCapture`` instances in the same order as input.

    Raises:
        ValueError: If any file cannot be opened.
    """
    return [load_video(path) for path in video_paths]
