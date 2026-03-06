"""
Tune the similarity_threshold used to label wide vs close-up frames.

Prints pixel-diff statistics for (result vs total_view) and (result vs closeup)
so you can visually identify a threshold that separates the two distributions.
Runs on the first concert in the config — the threshold is typically consistent
across concerts recorded with the same cameras.

Usage::

    python311 scripts/tune_threshold.py --config configs/model_config.yaml
"""

import argparse

import yaml

from src.data.sampler import compute_similarity_stats, generate_training_data


def main(config: dict) -> None:
    """Print similarity stats for both camera pairs and suggest a threshold."""
    # Use the first concert — threshold is camera-dependent, not concert-dependent
    concert = config["data"]["concerts"][0]
    total_view = concert["total_view"]
    closeup    = concert["closeup"]
    result     = concert["result"]

    print("Computing pixel-difference statistics (200 random samples)...")
    print("  Comparing result vs total_view  AND  result vs closeup ...\n")

    stats = compute_similarity_stats(
        total_view_path = total_view,
        closeup_path    = closeup,
        result_path     = result,
        n_samples       = 200,
    )

    for key, label in [("total", "result ↔ total_view"), ("closeup", "result ↔ closeup")]:
        s = stats[key]
        print(f"── {label} ──────────────────────────────")
        for k, v in s.items():
            print(f"  {k:>5}: {v:.2f}")
        print()

    # A good threshold sits between the median of the two distributions
    suggested = (stats["total"]["p50"] + stats["closeup"]["p50"]) / 2.0
    print(f"Suggested similarity_threshold: {suggested:.1f}")
    print("  (Wide frames:     result ≈ total_view  → diff < threshold)")
    print("  (Close-up frames: result ≈ closeup     → diff < threshold)")
    print("  Ambiguous frames in between are skipped during training.")

    # Quick label distribution check at suggested threshold
    print(f"\nChecking label split at threshold={suggested:.1f} ...")
    _, labels = generate_training_data(
        total_view_path      = total_view,
        closeup_path         = closeup,
        result_path          = result,
        sample_fps           = config["data"].get("sample_fps", 1.0),
        similarity_threshold = suggested,
        cache_dir            = config["data"].get("cache_dir", "data/cache"),
        force_rebuild        = True,   # always re-evaluate at the new threshold
    )
    n = len(labels)
    w = labels.count(0)
    c = labels.count(1)
    print(f"  Wide: {w} ({w / n * 100:.1f}%)  |  Close-up: {c} ({c / n * 100:.1f}%)")
    print("\nAdd to configs/model_config.yaml:")
    print(f"  similarity_threshold: {suggested:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune the pixel-diff similarity threshold."
    )
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
