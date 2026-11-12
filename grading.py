import numpy as np


def f_score(true_positives, n_predicted, n_true):
    """F = 2PR / (P + R). Returns 0 if nothing was correct."""
    if true_positives == 0:
        return 0.0
    precision = true_positives / n_predicted
    recall = true_positives / n_true
    return 2 * precision * recall / (precision + recall)


def match_spikes(true_index, pred_index, tolerance=50):
    """Pair each true spike with the closest unused prediction within +/- tolerance samples.
    Both inputs must be sorted. Returns the positions of the matched true spikes and predictions."""
    used = np.zeros(len(pred_index), dtype=bool)
    true_matched = []
    pred_matched = []
    for t in range(len(true_index)):
        lo = np.searchsorted(pred_index, true_index[t] - tolerance, side="left")
        hi = np.searchsorted(pred_index, true_index[t] + tolerance, side="right")
        best = None
        best_distance = tolerance + 1
        for p in range(lo, hi):
            distance = abs(pred_index[p] - true_index[t])
            if not used[p] and distance < best_distance:
                best = p
                best_distance = distance
        if best is not None:
            used[best] = True
            true_matched.append(t)
            pred_matched.append(best)
    return np.array(true_matched, dtype=int), np.array(pred_matched, dtype=int)


def grade(true_index, true_class, pred_index, pred_class, tolerance=50, class_weighting="count"):
    """F_total = 0.3 F_detect + 0.7 F_class.
    class_weighting: "count" weights each class by its number of true spikes, "equal" gives 20% each."""
    true_matched, pred_matched = match_spikes(true_index, pred_index, tolerance)

    # Detection: a hit is any matched pair
    f_detect = f_score(len(true_matched), len(pred_index), len(true_index))

    # Classification: per class, a hit is a matched pair where both classes equal c
    correct = true_class[true_matched] == pred_class[pred_matched]
    f_per_class = []
    for c in range(1, 6):
        tp = np.sum(correct & (true_class[true_matched] == c))
        f_per_class.append(float(f_score(tp, np.sum(pred_class == c), np.sum(true_class == c))))

    if class_weighting == "count":
        weights = [np.sum(true_class == c) for c in range(1, 6)]
    else:
        weights = [1, 1, 1, 1, 1]
    f_class = float(np.average(f_per_class, weights=weights))

    f_total = 0.3 * f_detect + 0.7 * f_class
    return {"f_total": f_total, "f_detect": f_detect, "f_class": f_class, "f_per_class": f_per_class}
