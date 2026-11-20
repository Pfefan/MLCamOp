# Concert Video Editor (MLCamOp)

Auto-cut between wide-shot and close-up concert cameras using a fine-tuned MobileNetV3-Small classifier.

## How it works

1. **Labelling** — For each concert you provide three synced videos: the wide shot, the close-up, and the human-edited result. The result is compared against both cameras to label every sampled frame as wide or close-up.
2. **Training** — A MobileNetV3-Small classifier is fine-tuned on those labels to learn which camera to show at any moment.
3. **Editing** — The model scores a new concert frame by frame, the predictions are smoothed into stable shots, and the result is rendered to an MP4.

## Project structure

```
configs/model_config.yaml   # Video paths, hyperparameters, model checkpoint path
scripts/
  train.py                  # Train the classifier
  evaluate.py               # Evaluate on the held-out concert(s)
  tune_threshold.py         # Calibrate pixel-diff threshold per concert
  preview.py                # Render an edited cut
src/
  data/sampler.py           # Frame sampling, labelling, caching
  models/view_classifier.py # MobileNetV3 classifier (train + predict)
  models/scene_detector.py  # Heuristic scene-change detector
  pipeline/inference.py     # Run classifier over a video
  pipeline/editing.py       # Assemble final cut from labels
  postprocessing/renderer.py# Write frames to MP4
  utils/logger.py           # Logging setup
  utils/metrics.py          # Evaluation metrics and shot smoothing
  utils/visualization.py    # Debug overlays
tests/                      # Unit tests
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

All scripts read `configs/model_config.yaml` by default (concert video paths, sampling rate, model path, and training settings).

```bash
# Calibrate the pixel-diff threshold for a concert (optional, once per concert)
python scripts/tune_threshold.py --concert 0

# Train
python scripts/train.py

# Evaluate on the held-out concert(s)
python scripts/evaluate.py

# Render an edited cut of a concert from the config
python scripts/preview.py --concert 0 --output output/result.mp4

# ...or render from explicit video paths
python scripts/preview.py \
    --input path/to/wide.mp4 --closeup path/to/closeup.mp4 \
    --output output/preview.mp4
```

## Configuration

Everything is driven by `configs/model_config.yaml`: concert video paths, sampling rate, training hyperparameters, and the model checkpoint path.

One knob worth knowing is the cut sensitivity under `evaluation.smoothing` — raise it for fewer, longer shots, lower it for more frequent cuts. It only affects editing, so you can tune it without retraining.

### Adding a concert

Append an entry with a unique `id` and the three video paths to `data.concerts`, then run `python scripts/train.py`. The first run decodes and caches the new concert; later runs memory-map the cache.

```yaml
data:
  concerts:
    - id: my_new_concert
      total_view: "/path/to/wide.mp4"
      closeup: "/path/to/closeup.mp4"
      result: "/path/to/edit.mp4"
  val_concerts: ["ubahnorchester"]
```

`data.val_concerts` lists the ids held out for validation; every other concert is trained on, so the position of an entry in the list does not matter. Scores are averaged per held-out concert. Keep the held-out set fixed when comparing runs, and hold out more than one concert once enough are available — a single concert is a noisy yardstick.

Training weights samples so that every concert, and both classes inside each concert, contribute equally. A long concert or one with an unusual wide/close-up ratio therefore cannot dominate.

Training progress (one line per epoch) is written to the console and to `logging.log_file`.

## Memory requirements

Training fits in **32 GB RAM** — frames are stored as uint8 numpy arrays (~150 KB each) and preprocessed on-the-fly during training. The cache uses stacked tensors for efficient I/O.

## Tests

```bash
python -m pytest tests/ -v
```
