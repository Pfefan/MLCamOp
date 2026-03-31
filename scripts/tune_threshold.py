"""Tune the similarity_threshold for labelling wide vs close-up frames.

Each concert can have its own threshold. Run once per concert and set the
suggested value in the corresponding field in model_config.yaml.

Usage::

    python scripts/tune_threshold.py --config configs/model_config.yaml
    python scripts/tune_threshold.py --config configs/model_config.yaml --concert 1 --check-split
"""

import argparse

import yaml

from src.data.sampler import compute_similarity_stats, generate_training_data


def main(config: dict, concert_index: int = 0, check_split: bool = False) -> None:
    """Print similarity stats for one concert and suggest a threshold."""
    concerts = config["data"]["concerts"]
    if concert_index >= len(concerts):
        raise IndexError(
            f"--concert {concert_index} is out of range "
            f"({len(concerts)} concert(s) defined in config)."
        )

    concert    = concerts[concert_index]
    total_view = concert["total_view"]
    closeup    = concert["closeup"]
    result     = concert["result"]

    print(f"Concert {concert_index + 1}/{len(concerts)}: {total_view}")
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

    # Warn if the two distributions are identical — usually means a path problem
    if abs(stats["total"]["p50"] - stats["closeup"]["p50"]) < 0.01:
        print("⚠️  WARNING: result↔total_view and result↔closeup distributions are")
        print("   identical. This usually means both video paths point to the same file.")
        print(f"   total_view : {total_view}")
        print(f"   closeup    : {closeup}")
        print()

    print(f"Suggested similarity_threshold: {suggested:.1f}")
    print("  (Wide frames:     result ≈ total_view  → diff < threshold)")
    print("  (Close-up frames: result ≈ closeup     → diff < threshold)")
    print("  Ambiguous frames in between are skipped during training.")
    print(f"\nUpdate concert {concert_index + 1} in configs/model_config.yaml:")
    print(f"  similarity_threshold: {suggested:.1f}")

    if check_split:
        # Full video sampling — slow but confirms the threshold is sensible
        print(f"\nChecking label split at threshold={suggested:.1f} (this may take a few minutes)...")
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
    else:
        print("\n(Add --check-split to also verify the label split across the full video.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune the pixel-diff similarity threshold for one concert."
    )
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument(
        "--concert",
        type=int,
        default=0,
        help="0-based index of the concert to tune (default: 0).",
    )
    parser.add_argument(
        "--check-split",
        action="store_true",
        help="Also verify the label split across the full video (slow).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg, concert_index=args.concert, check_split=args.check_split)
