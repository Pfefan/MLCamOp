import sys
from src.pipeline.ingestion import ingest_videos
from src.pipeline.inference import run_inference
from src.pipeline.editing import edit_video
from src.postprocessing.renderer import render_video
from src.config.settings import VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH

def main():
    # Ingest video data
    video_data = ingest_videos(VIDEO_INPUT_PATH)

    # Run inference to classify views and detect scenes
    processed_data = run_inference(video_data)

    # Edit the video based on the processed data
    edited_video = edit_video(processed_data)

    # Render the final edited video output
    render_video(edited_video, VIDEO_OUTPUT_PATH)

if __name__ == "__main__":
    main()