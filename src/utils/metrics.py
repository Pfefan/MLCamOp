"""Metrics for label and cut-event evaluation."""

import numpy as np


def cut_indices(labels: list[int]) -> list[int]:
    """Return indices where the label changes."""
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def shot_lengths(labels: list[int]) -> list[int]:
    """Return the length (in frames) of every contiguous same-label run."""
    if not labels:
        return []
    lengths: list[int] = []
    run = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            run += 1
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return lengths


def pacing_stats(labels: list[int], sample_fps: float) -> dict:
    """Cut rate and shot-duration distribution for a label sequence."""
    lengths = shot_lengths(labels)
    n_cuts  = len(cut_indices(labels))
    minutes = len(labels) / sample_fps / 60.0 if labels else 0.0
    durations = np.array(lengths, dtype=np.float64) / sample_fps if lengths else np.zeros(0)

    return {
        "n_shots":        len(lengths),
        "n_cuts":         n_cuts,
        "cuts_per_min":   n_cuts / minutes if minutes > 0 else 0.0,
        "median_shot_s":  float(np.median(durations)) if durations.size else 0.0,
        "p10_shot_s":     float(np.percentile(durations, 10)) if durations.size else 0.0,
        "p90_shot_s":     float(np.percentile(durations, 90)) if durations.size else 0.0,
        "min_shot_s":     float(durations.min()) if durations.size else 0.0,
    }


def flicker_count(labels: list[int], min_shot_frames: int) -> int:
    """Number of shots shorter than *min_shot_frames* (jitter indicator)."""
    return sum(1 for length in shot_lengths(labels) if length < min_shot_frames)


def onset_envelope(audio: np.ndarray) -> np.ndarray:
    """Spectral-flux onset strength per frame from log-mel features.

    Positive frame-to-frame energy increases summed across mel bands; peaks
    line up with note/beat attacks, giving a cheap musical-timing reference
    without a full beat tracker.
    """
    if audio is None or len(audio) < 2:
        return np.zeros(len(audio) if audio is not None else 0)
    diff = np.diff(audio, axis=0)
    flux = np.maximum(diff, 0.0).sum(axis=1)
    env  = np.concatenate([[0.0], flux])
    peak = env.max()
    return env / peak if peak > 0 else env


def cut_onset_alignment(
    labels:     list[int],
    audio:      np.ndarray,
    sample_fps: float,
    percentile: float = 75.0,
) -> dict:
    """How close predicted cuts land to musical onsets.

    Onset frames are flux peaks above *percentile*; for each cut we take the
    distance (in seconds) to the nearest onset frame. Lower = more musical.
    """
    cuts = cut_indices(labels)
    env  = onset_envelope(audio)
    if not cuts or env.size == 0:
        return {"n_cuts": len(cuts), "median_dist_s": None, "frac_on_beat": None}

    threshold   = np.percentile(env, percentile)
    onset_idx   = np.where(env >= threshold)[0]
    if onset_idx.size == 0:
        return {"n_cuts": len(cuts), "median_dist_s": None, "frac_on_beat": None}

    cut_arr     = np.array(cuts)
    nearest_pos = np.searchsorted(onset_idx, cut_arr)
    nearest_pos = np.clip(nearest_pos, 1, len(onset_idx) - 1)
    left        = onset_idx[nearest_pos - 1]
    right       = onset_idx[nearest_pos]
    dist_frames = np.minimum(np.abs(cut_arr - left), np.abs(cut_arr - right))
    dist_s      = dist_frames / sample_fps

    on_beat_tol = 1.0 / sample_fps  # within one sample step
    return {
        "n_cuts":        len(cuts),
        "median_dist_s": float(np.median(dist_s)),
        "frac_on_beat":  float(np.mean(dist_s <= on_beat_tol)),
    }


def balanced_accuracy(gt: list[int], pred: list[int]) -> float:
    """Mean per-class recall."""
    classes = set(gt)
    if not classes:
        return 0.0
    recalls = []
    for c in classes:
        total = sum(1 for g in gt if g == c)
        hit = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        recalls.append(hit / total if total else 0.0)
    return sum(recalls) / len(recalls)


def cut_event_prf(gt: list[int], pred: list[int], tolerance: int) -> dict:
    """Precision/recall/F1 for cut events within a tolerance window."""
    gt_cuts = cut_indices(gt)
    pred_cuts = cut_indices(pred)

    matched: set[int] = set()
    true_positives = 0
    for p in pred_cuts:
        best, best_dist = None, tolerance + 1
        for j, g in enumerate(gt_cuts):
            if j in matched:
                continue
            dist = abs(p - g)
            if dist <= tolerance and dist < best_dist:
                best, best_dist = j, dist
        if best is not None:
            true_positives += 1
            matched.add(best)

    false_pos = len(pred_cuts) - true_positives
    false_neg = len(gt_cuts) - true_positives
    precision = true_positives / (true_positives + false_pos) if pred_cuts else 0.0
    recall    = true_positives / (true_positives + false_neg) if gt_cuts else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "n_gt":      len(gt_cuts),
        "n_pred":    len(pred_cuts),
        "tp":        true_positives,
    }


def viterbi_smooth(probs: np.ndarray, switch_penalty: float, min_shot: int = 1) -> list[int]:
    """Decode the most likely label path from per-frame class probabilities.

    HMM Viterbi where emission = log P(class) and every camera switch costs
    *switch_penalty* log-units. Higher penalty => longer shots / fewer cuts.
    A penalty of 0 reduces to plain per-frame argmax.

    ``min_shot`` > 1 adds a hard constraint: no switch is allowed until the
    current shot has lasted at least *min_shot* samples (dwell-expanded HMM).
    """
    probs = np.asarray(probs, dtype=np.float64)
    n, k = probs.shape
    if n == 0:
        return []
    if min_shot > 1:
        return _viterbi_min_shot(probs, switch_penalty, min_shot)
    if switch_penalty <= 0.0:
        return probs.argmax(axis=1).astype(int).tolist()

    emission = np.log(np.clip(probs, 1e-8, 1.0))
    dp   = emission[0].copy()
    back = np.zeros((n, k), dtype=np.int64)

    for t in range(1, n):
        # For each current state, pick the best previous state under the
        # switch cost (0 to stay, -penalty to change).
        scores    = dp[:, None] - switch_penalty * (1 - np.eye(k))  # (prev, cur)
        best_from = scores.argmax(axis=0)
        prev_best = scores[best_from, np.arange(k)]
        dp = emission[t] + prev_best
        back[t] = best_from

    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(dp.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path.astype(int).tolist()


def _viterbi_min_shot(probs: np.ndarray, switch_penalty: float, min_shot: int) -> list[int]:
    """Viterbi over dwell-expanded states (class, time-in-shot capped at min_shot-1).

    Switching into another class is only reachable from the max-dwell state, so
    every emitted shot is guaranteed to span at least *min_shot* samples.
    """
    n, k = probs.shape
    D = int(min_shot)
    emission = np.log(np.clip(probs, 1e-8, 1.0))

    dp = np.full((k, D), -np.inf)
    dp[:, 0] = emission[0]
    back = np.zeros((n, k, D), dtype=np.int32)  # encodes prev state as c*D+d

    for t in range(1, n):
        new    = np.full((k, D), -np.inf)
        new_bk = np.zeros((k, D), dtype=np.int32)
        for c in range(k):
            # Stay in the same class: dwell advances (capped at D-1).
            for d in range(D):
                nd = min(d + 1, D - 1)
                if dp[c, d] > new[c, nd]:
                    new[c, nd]    = dp[c, d]
                    new_bk[c, nd] = c * D + d
            # Switch: only allowed once the previous shot reached full dwell.
            for pc in range(k):
                if pc == c:
                    continue
                score = dp[pc, D - 1] - switch_penalty
                if score > new[c, 0]:
                    new[c, 0]    = score
                    new_bk[c, 0] = pc * D + (D - 1)
        dp      = new + emission[t][:, None]
        back[t] = new_bk

    state = int(dp.argmax())
    path  = np.zeros(n, dtype=np.int64)
    path[-1] = state // D
    for t in range(n - 1, 0, -1):
        state    = int(back[t, state // D, state % D])
        path[t - 1] = state // D
    return path.astype(int).tolist()


def calibrate_switch_penalty(
    probs:        np.ndarray,
    target_cuts:  int,
    lo:           float = 0.0,
    hi:           float = 25.0,
    iterations:   int   = 30,
    min_shot:     int   = 1,
) -> float:
    """Bisection search for the penalty whose smoothed path has ~target_cuts.

    Cut count decreases monotonically with penalty, so a binary search over
    the penalty lands close to the desired pacing (e.g. the GT cut count).
    """
    def n_cuts(pen: float) -> int:
        return len(cut_indices(viterbi_smooth(probs, pen, min_shot=min_shot)))

    if n_cuts(lo) <= target_cuts:
        return lo
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if n_cuts(mid) > target_cuts:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
