"""Score spike guesses against the real spikes (detection only, no classes yet).

Usage from another script (run from the project root):
    sys.path.insert(0, "Modular")
    from f_score import f_detect
    result = f_detect(real_index, guesses)    # result["f"], result["precision"], ...
"""
import numpy as np


def match(real, guesses, tolerance=50):
    """Pair real spikes with guesses, closest pairs first.

    Every (real, guess) pair within ±tolerance samples is a candidate. The
    candidates are taken in order of distance, smallest first, and a pair is
    kept only if neither its real spike nor its guess has been used yet.
    So each real spike gets at most one guess, and each guess at most one spike.

    Returns two arrays of the same length: positions in `real` and positions
    in `guesses` that were paired.
    """
    real = np.asarray(real)
    guesses = np.asarray(guesses)
    order = np.argsort(guesses)            # searchsorted needs sorted guesses
    sorted_guesses = guesses[order]

    # every candidate pair within tolerance
    pairs = []                             # (distance, real position, guess position)
    for r, spike in enumerate(real):
        lo = np.searchsorted(sorted_guesses, spike - tolerance, side="left")
        hi = np.searchsorted(sorted_guesses, spike + tolerance, side="right")
        for k in range(lo, hi):
            pairs.append((abs(int(sorted_guesses[k]) - int(spike)), r, int(order[k])))
    pairs.sort()

    real_used = np.zeros(len(real), dtype=bool)
    guess_used = np.zeros(len(guesses), dtype=bool)
    real_matched, guess_matched = [], []
    for _, r, g in pairs:
        if not real_used[r] and not guess_used[g]:
            real_used[r] = guess_used[g] = True
            real_matched.append(r)
            guess_matched.append(g)
    return np.array(real_matched, dtype=int), np.array(guess_matched, dtype=int)


def f_detect(real, guesses, tolerance=50):
    """Detection score.

    precision = hits / number of guesses   (how many guesses were right)
    recall    = hits / number of real spikes (how many spikes were found)
    F         = 2PR / (P + R)
    """
    real_matched, guess_matched = match(real, guesses, tolerance)
    hits, n_guesses, n_real = len(real_matched), len(guesses), len(real)

    precision = hits / n_guesses if n_guesses else 0.0
    recall = hits / n_real if n_real else 0.0
    f = 2 * precision * recall / (precision + recall) if hits else 0.0

    return {"f": f, "precision": precision, "recall": recall, "hits": hits,
            "guesses": n_guesses, "real": n_real,
            "real_matched": real_matched, "guess_matched": guess_matched}
