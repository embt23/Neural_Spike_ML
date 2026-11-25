"""Matched filter bank vs the current pipeline (the 2026-11-24 submission's
"old" pipeline: all-spike detector + 6-way classifier), on validation data.

Validation signals are D1_train[SPLIT:] (never used for training by either
method) + generated noise. Two separate sets of noise seeds:
  tune:  N_TUNE signals per level, to choose the matched filter's threshold
  score: N_SCORE other signals per level, to score both methods
The old pipeline keeps its own thresholds. The test set is not touched.

Scored with the grading formula: F_detect (all spikes), F_class (mean F over
the 5 classes), F_total = 0.3 F_detect + 0.7 F_class. ±50 samples.

Run from the project root (levels optional, default 2-6):
    python Matched_filter/compare.py
Saves Matched_filter/data/compare.json, Matched_filter/data/thresholds.json
and Artifacts/Matched_Filter_Compare.png.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "Matched_filter")
sys.path.insert(0, "1detector-per-class")
from matched_filter import build_bank, SPLIT, CLASSES     # noqa: E402
from generate_noise import generate_noise                 # noqa: E402
from test_compare import old_guesses                      # noqa: E402
from main import count_hits                               # noqa: E402

N_TUNE, TUNE_SEED = 3, 7000      # seed = TUNE_SEED + 10 * level + signal number
N_SCORE, SCORE_SEED = 5, 8000
N_THRESHOLDS = 41                # thresholds tried, from the tune signals' peak scores
OUT_DIR = "Matched_filter/data"
CHART = "Artifacts/Matched_Filter_Compare.png"


def f(real, guesses):
    hits = count_hits(real, guesses) if len(guesses) else 0
    return 2 * hits / (len(real) + len(guesses)) if hits else 0.0     # = 2PR/(P+R)


def grade(real_i, real_c, i, c):
    """F_detect, F_class, F_total."""
    f_detect = f(real_i, i)
    f_class = np.mean([f(real_i[real_c == k], i[c == k]) for k in CLASSES])
    return np.array([f_detect, f_class, 0.3 * f_detect + 0.7 * f_class])


def signals(level, seed, n):
    t = np.load("Data/D1_train.npz")
    d = t["d"]
    return [d[SPLIT:] + generate_noise(level, len(d), seed=seed + 10 * level + s)[SPLIT:]
            for s in range(n)]


def main():
    levels = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6]
    os.makedirs(OUT_DIR, exist_ok=True)
    t = np.load("Data/D1_train.npz")
    val = t["Index"] >= SPLIT
    real_i, real_c = t["Index"][val] - SPLIT, t["Class"][val]

    results, thresholds = {}, {}
    print("level   method              F_detect  F_class  F_total")
    for level in levels:
        bank = build_bank(level)
        tune = signals(level, TUNE_SEED, N_TUNE)
        score = signals(level, SCORE_SEED, N_SCORE)

        # thresholds to try: spread over the scores of the top 4000 peaks
        peaks = np.sort(np.concatenate([bank.detect(x)[2] for x in tune]))[-4000:]
        tries = np.quantile(peaks, np.linspace(0, 1, N_THRESHOLDS))
        tune_f = [np.mean([grade(real_i, real_c, *bank.peel(x, th)) for x in tune], axis=0)[2]
                  for th in tries]
        th = float(tries[int(np.argmax(tune_f))])
        thresholds[level] = th

        rows = {
            "current pipeline": [grade(real_i, real_c, *keep_labelled(*old_guesses(level, x)))
                                 for x in score],
            "matched filter": [grade(real_i, real_c, *bank.detect(x, th)[:2]) for x in score],
            "matched filter + peel": [grade(real_i, real_c, *bank.peel(x, th)) for x in score],
        }
        results[level] = {name: np.mean(r, axis=0).round(4).tolist() for name, r in rows.items()}
        for name, (fd, fc, ft) in results[level].items():
            print(f"  {level}     {name:22s} {fd:.3f}    {fc:.3f}    {ft:.3f}", flush=True)

    with open(f"{OUT_DIR}/compare.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open(f"{OUT_DIR}/thresholds.json", "w") as fh:
        json.dump(thresholds, fh, indent=2)
    chart(results)
    print(f"\nSaved {OUT_DIR}/compare.json, {OUT_DIR}/thresholds.json and {CHART}")


def keep_labelled(i, c):
    """The old classifier's label 0 means 'not a spike': drop those."""
    return i[c != 0], c[c != 0]


def chart(results):
    names = ["current pipeline", "matched filter", "matched filter + peel"]
    colours = ["#9aa0a6", "#2a78d6", "#eb6834"]
    levels = list(results)
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (name, colour) in enumerate(zip(names, colours)):
        vals = [results[lv][name][2] for lv in levels]
        bars = ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=colour, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"Level {lv}" for lv in levels])
    ax.set_ylabel("F_total (validation)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Matched filter bank vs current pipeline")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHART, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
