"""Entry point for the Concert Video Editor pipeline."""

from src.config.settings import VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH
from src.pipeline.ingestion import ingest_video_data
from src.pipeline.inference import run_inference
from src.pipeline.editing import process_video
from src.postprocessing.renderer import render_final_video
from src.utils.logger import setup_logger


def main():
    """Run the full video processing pipeline."""
    logger = setup_logger()

    logger.info("Ingesting video data...")
    video_data = ingest_video_data([VIDEO_INPUT_PATH])

    logger.info("Running inference on video data...")
    processed_data = run_inference(video_data, model=None)

    logger.info("Editing video...")
    # process_video now returns the frame list
    final_frames = process_video([], [], [], [], VIDEO_OUTPUT_PATH)

    logger.info("Rendering final video...")
    render_final_video(final_frames, [], VIDEO_OUTPUT_PATH)

    logger.info("Video processing completed successfully.")
    _ = processed_data  # Will be used once model is trained


if __name__ == "__main__":
    main()