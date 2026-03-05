"""Configuration settings for the concert video editor project."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LOCATION = "E:/Job/Projekte/CutStudio/03_StadtorchesterJan2026"

# Paths
VIDEO_INPUT_PATH = os.path.join(DATA_LOCATION, "input")
VIDEO_OUTPUT_PATH = os.path.join(DATA_LOCATION, "output")
MODEL_PATH = os.path.join(BASE_DIR, "models")
LOGGING_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "logging_config.yaml")

# Model parameters
VIEW_CLASSIFIER_MODEL = os.path.join(MODEL_PATH, "view_classifier.h5")
SCENE_DETECTOR_MODEL = os.path.join(MODEL_PATH, "scene_detector.h5")

# Processing parameters
FRAME_RATE = 30
RESOLUTION = (1920, 1080)  # Width, Height
SHOT_SCORING_THRESHOLD = 0.5

# Logging settings
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Other settings
ENABLE_AUDIO_ANALYSIS = True
ENABLE_TRANSITIONS = True