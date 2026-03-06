"""
Project-wide constants derived from the YAML configs.

All mutable settings (paths, thresholds, hyperparameters) live in
``configs/model_config.yaml`` and are loaded at runtime by the scripts.
This module only holds the truly static constants that never change between
concerts: default FPS, codec, resolution cap, etc.

To load the full config in a script::

    import yaml
    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)
"""

# ── Output encoding ───────────────────────────────────────────────────────────
OUTPUT_CODEC    = "mp4v"          # OpenCV fourcc codec string
DEFAULT_FPS     = 25.0            # fallback when source FPS cannot be read
MAX_RESOLUTION  = (3840, 2160)    # 4K upper bound; resize if exceeded

# ── Training defaults (overridden by model_config.yaml) ──────────────────────
DEFAULT_SAMPLE_FPS           = 1.0
DEFAULT_SIMILARITY_THRESHOLD = 65.9
DEFAULT_BATCH_SIZE           = 64
DEFAULT_EPOCHS               = 40

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL    = "INFO"
DEFAULT_LOG_FILE = "logs/app.log"
