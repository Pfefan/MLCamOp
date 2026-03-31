"""Debug visualization helpers for inspecting frame predictions."""

import cv2
import numpy as np

from src.models.view_classifier import ViewClassifier


def visualize_frame(
    frame:       np.ndarray,
    label:       int,
    confidence:  float | None = None,
    title:       str          = "Frame",
) -> None:
    """Overlay predicted label on a frame and display in an OpenCV window."""
    output = frame.copy()
    label_text = "wide" if label == 0 else "close-up"
    text = f"{label_text} ({confidence:.2f})" if confidence is not None else label_text
    color = (0, 200, 0) if label == 0 else (0, 80, 255)   # green = wide, orange = close-up
    cv2.putText(output, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow(title, output)
    cv2.waitKey(1)


def visualize_video(
    video_path:     str,
    classifier:     ViewClassifier,
    frame_interval: int = 30,
) -> None:
    """Play a video in a window with per-frame label overlays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = 0
    current_label = 0   # hold last known label between sample points

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            current_label = classifier.predict(frame)

        visualize_frame(frame, current_label, title="Preview")
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
