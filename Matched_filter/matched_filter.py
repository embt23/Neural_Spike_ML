"""Whitened matched filter bank: one detector per class, no training.

Idea: we know what each class's spike looks like (its average shape, the
"template"), and we know what the noise looks like (its covariance: how
noise samples move together). For a known shape in Gaussian noise, the best
possible linear detector is the *whitened matched filter* (Franke et al.
2015, "Bayes optimal template matching for spike sorting").

For class c, at every sample position:
    score_c = w_c . window - E_c / 2
  window = the signal from PRE samples before that position to POST after
  w_c    = C^-1 t_c   (t_c = class c's template, C = noise covariance)
  E_c    = t_c . w_c  (how strong class c is compared with the noise)

score_c is the log-likelihood ratio "a class c spike starts here" vs "just
noise". Multiplying by C^-1 is the "whitening": it turns down the directions
where the noise is big (slow drift, at these noise levels) and turns up the
ones where it is small. The window . w_c part is the "matched filter": the
template, slid along the signal (the same as filtering with the template
reversed in time).

Detection: take the best class score at every position, keep local peaks
(the highest within ±RADIUS samples) above a threshold. Class = the class
with the best score at that peak.

Peeling (for overlapping spikes): after detecting, subtract each detected
spike's template from the signal and detect again in what's left. A spike
hidden under a bigger neighbour can then show up. Repeat up to ROUNDS times.

Everything is built from the training part of D1 only (D1_train[:SPLIT]):
templates from its spikes, noise covariance from it + generated noise.

Usage from another script (run from the project root):
    sys.path.insert(0, "Matched_filter")
    from matched_filter import build_bank
    bank = build_bank(level=6)
    index, cls = bank.peel(signal, threshold)     # 0-based spike starts, classes 1-5
"""
import sys
from functools import lru_cache

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import correlate

sys.path.insert(0, "Modular")
from generate_noise import generate_noise    # noqa: E402

SPLIT = 1_000_000          # templates and noise come from D1_train[:SPLIT] only
PRE, POST = 50, 200        # window: 2 ms before the spike start to 8 ms after
RADIUS = 20                # a detection must be the highest score within ±RADIUS
ROUNDS = 4                 # peeling rounds
RIDGE = 1e-3               # small boost to C's diagonal so C^-1 is stable
N_NOISE = 20000            # noise windows used to measure the covariance
GAP_BEFORE, GAP_AFTER = 50, 120   # samples near a spike that don't count as noise
NOISE_SEED = 99            # generated noise used to measure the covariance
CLASSES = np.array([1, 2, 3, 4, 5])


def cut(x, starts):
    """One row per start: x from PRE before it to POST after. Starts too close
    to either end are skipped."""
    starts = starts[(starts >= PRE) & (starts + POST <= len(x))]
    return np.stack([x[s - PRE:s + POST] for s in starts])


@lru_cache(maxsize=None)
def training_data():
    """D1_train[:SPLIT] (median removed) and its spikes."""
    t = np.load("Data/D1_train.npz")
    d, index, cls = t["d"][:SPLIT], t["Index"], t["Class"]
    keep = index < SPLIT
    return d - np.median(d), index[keep], cls[keep]


def templates():
    """Average spike of each class (rows of a (5, PRE + POST) array), with the
    baseline (mean of the window before the spike) subtracted."""
    d, index, cls = training_data()
    rows = []
    for c in CLASSES:
        mean = cut(d, index[cls == c]).mean(axis=0)
        rows.append(mean - mean[:PRE - 2].mean())
    return np.stack(rows)


def noise_covariance(level, seed=NOISE_SEED):
    """Covariance of (PRE + POST)-sample windows of pure noise: D1_train[:SPLIT]
    plus generated level noise, at random spots with no spike nearby."""
    d, index, _ = training_data()
    x = d + generate_noise(level, len(d), seed=seed)
    rng = np.random.default_rng(0)
    starts = rng.integers(PRE, len(x) - POST, size=20 * N_NOISE)   # extra, most get dropped
    lo = np.searchsorted(index, starts - PRE - GAP_AFTER)
    hi = np.searchsorted(index, starts + POST + GAP_BEFORE)
    windows = cut(x, starts[lo == hi][:N_NOISE])                  # equal = no spike inside
    C = np.cov(windows, rowvar=False)
    return C + RIDGE * np.trace(C) / len(C) * np.eye(len(C))


class MatchedFilterBank:
    def __init__(self, T, C):
        self.T = T                                   # templates, (5, L)
        self.W = np.linalg.solve(C, T.T).T           # whitened filters w_c = C^-1 t_c
        self.E = np.sum(T * self.W, axis=1)          # E_c = t_c . w_c

    def scores(self, x):
        """Log-likelihood ratio of each class at every window start, (5, len(x) - L + 1).
        Column j is about a spike starting at sample j + PRE."""
        x = x - np.median(x)
        return np.stack([correlate(x, w, mode="valid", method="fft") for w in self.W]) \
            - self.E[:, None] / 2

    def detect(self, x, threshold=-np.inf):
        """Spike starts (0-based), classes and scores of every local peak above threshold."""
        s = self.scores(x)
        best = s.max(axis=0)
        j = np.flatnonzero((maximum_filter1d(best, 2 * RADIUS + 1) == best) & (best > threshold))
        return j + PRE, CLASSES[s[:, j].argmax(axis=0)], best[j]

    def peel(self, x, threshold, rounds=ROUNDS):
        """detect(), subtract the detected templates, detect again in what's left.
        Returns spike starts (0-based, sorted) and classes."""
        residual = np.array(x, dtype=np.float64)
        residual -= np.median(residual)
        found_i, found_c = [], []
        for _ in range(rounds):
            i, c, _ = self.detect(residual, threshold)
            if len(i) == 0:
                break
            for start, k in zip(i, c):
                residual[start - PRE:start + POST] -= self.T[k - 1]
            found_i.append(i)
            found_c.append(c)
        if not found_i:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        i, c = np.concatenate(found_i), np.concatenate(found_c)
        order = np.argsort(i)
        return i[order], c[order]


@lru_cache(maxsize=None)
def build_bank(level):
    """The filter bank for one noise level (2-6). Cached."""
    return MatchedFilterBank(templates(), noise_covariance(level))
