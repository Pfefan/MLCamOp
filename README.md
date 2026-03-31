# Concert Video Editor (MLCamOp)

Auto-cut between wide-shot and close-up concert cameras using a fine-tuned MobileNetV3-Small classifier.

## How it works

1. **Labelling** — Given three synchronised videos per concert (wide, close-up, human-edited result), the sampler compares each result frame against both cameras via pixel-diff to generate `wide (0)` / `close-up (1)` labels.
2. **Training** — MobileNetV3-Small is fine-tuned on those labelled frames with differential learning rates, class weighting, augmentation, AMP, and early stopping.
3. **Inference** — The trained model classifies frames of a new wide-shot video; a short-shot filter smooths the predictions.
4. **Assembly** — Frames are picked from wide or close-up based on the predictions and rendered to MP4.

## Project structure

```
configs/model_config.yaml   # Video paths, hyperparameters, model checkpoint path
scripts/
  train.py                  # Train the classifier
  evaluate.py               # Evaluate on the held-out concert
  tune_threshold.py         # Calibrate pixel-diff threshold per concert
  run_pipeline.py           # End-to-end inference on a concert
  preview.py                # Quick preview with optional digital-zoom close-up
src/
  data/sampler.py           # Frame sampling, labelling, caching
  models/view_classifier.py # MobileNetV3 classifier (train + predict)
  models/scene_detector.py  # Heuristic scene-change detector
  pipeline/inference.py     # Run classifier over a video
  pipeline/editing.py       # Assemble final cut from labels
  postprocessing/renderer.py# Write frames to MP4
  utils/logger.py           # Logging setup
  utils/visualization.py    # Debug overlays
tests/                      # Unit tests
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Calibrate thresholds (once per concert)
python scripts/tune_threshold.py --config configs/model_config.yaml --concert 0

# 2. Train
python scripts/train.py --config configs/model_config.yaml

# 3. Evaluate
python scripts/evaluate.py --config configs/model_config.yaml

# 4. Run on a concert
python scripts/run_pipeline.py --config configs/model_config.yaml --output output/result.mp4

# 5. Quick preview (single wide-shot, digital zoom as close-up)
python scripts/preview.py --config configs/model_config.yaml \
    --input "path/to/wide.mp4" --output output/preview.mp4
```

## Memory requirements

Training fits in **32 GB RAM** — frames are stored as uint8 numpy arrays (~150 KB each) and preprocessed on-the-fly during training. The cache uses stacked tensors for efficient I/O.

## Tests

```bash
python -m pytest tests/ -v
```
