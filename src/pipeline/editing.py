"""Pipeline stage: assemble final video from total view and close-up frames."""

import cv2
import numpy as np


def edit_video(
    total_view_frames: list[np.ndarray],
    close_up_frames: list[np.ndarray],
    scene_changes: list[int],
) -> list[np.ndarray]:
    """
    Interleave total-view and close-up frames at detected scene change points.

    Args:
        total_view_frames: Frames from the wide/total camera.
        close_up_frames: Frames from the close-up camera.
        scene_changes: Frame indices where a cut to close-up should occur.

    Returns:
        List of frames for the edited video.
    """
    edited_video = []

    for i, frame in enumerate(total_view_frames):
        edited_video.append(frame)
        if i < len(scene_changes) and scene_changes[i] < len(close_up_frames):
            edited_video.append(close_up_frames[scene_changes[i]])

    return edited_video


def apply_transitions(
    edited_video: list[np.ndarray],
    transition_effects: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Insert transition frames between cuts in the edited video.

    Args:
        edited_video: Sequence of video frames.
        transition_effects: Frames to insert as transitions.

    Returns:
        Video frames with transitions inserted.
    """
    final_video = []

    for i in range(len(edited_video) - 1):
        final_video.append(edited_video[i])
        if i < len(transition_effects):
            final_video.append(transition_effects[i])

    if edited_video:
        final_video.append(edited_video[-1])

    return final_video


def save_edited_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 30.0,
) -> None:
    """
    Write assembled frames to a video file.

    Args:
        frames: List of frames to write.
        output_path: Destination file path.
        fps: Frames per second for output video.
    """
    if not frames:
        raise ValueError("Cannot save empty frame list.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()


def process_video(
    total_view_frames: list[np.ndarray],
    close_up_frames: list[np.ndarray],
    scene_changes: list[int],
    transition_effects: list[np.ndarray],
    output_path: str,
) -> list[np.ndarray]:
    """
    Full editing pipeline: edit, add transitions, and save.

    Args:
        total_view_frames: Wide-shot frames.
        close_up_frames: Close-up frames.
        scene_changes: Indices of scene cut points.
        transition_effects: Transition frames to insert.
        output_path: Path to write the final video.

    Returns:
        The final assembled list of frames.
    """
    edited = edit_video(total_view_frames, close_up_frames, scene_changes)
    final = apply_transitions(edited, transition_effects)
    save_edited_video(final, output_path)
    return final