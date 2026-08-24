"""Metrics for label and cut-event evaluation."""


def cut_indices(labels: list[int]) -> list[int]:
    """Return indices where the label changes."""
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


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
