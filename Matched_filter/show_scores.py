"""Picture of the matched filter bank at work on a short stretch of signal.

Three panels, same time axis:
  1. clean D1 (validation part), real spikes marked by class
  2. the same stretch + level noise (what the detector sees)
  3. each class's score (log-likelihood ratio), the threshold, and the
     detections after peeling, coloured by the class they were given

Run from the project root (level and start sample optional):
    python Matched_filter/show_scores.py 6 52000
Saves Artifacts/Matched_Filter_Scores_L<level>.png.
"""
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "Matched_filter")
sys.path.insert(0, "Other")
from matched_filter import build_bank, SPLIT, PRE                # noqa: E402
from generate_noise import generate_noise                       # noqa: E402
from composite_spikes import COLOURS                            # noqa: E402

FS = 25000
SPAN = 1500                    # samples shown (60 ms)
SEED = 8060                    # one of compare.py's scoring signals at level 6


def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 52000
    t = np.load("Data/D1_train.npz")
    clean = t["d"][SPLIT:]
    keep = t["Index"] >= SPLIT
    real_i, real_c = t["Index"][keep] - SPLIT, t["Class"][keep]
    noisy = clean + generate_noise(level, len(t["d"]), seed=SEED)[SPLIT:]
    with open("Matched_filter/data/thresholds.json") as fh:
        threshold = json.load(fh)[str(level)]

    bank = build_bank(level)
    scores = bank.scores(noisy)                                  # column j = spike start j + PRE
    found_i, found_c = bank.peel(noisy, threshold)

    end = start + SPAN
    ms = lambda s: (np.asarray(s) - start) / FS * 1000          # sample -> ms in the window
    window = np.arange(start, end)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(ms(window), clean[start:end], color="#333333", linewidth=0.8)
    axes[1].plot(ms(window), noisy[start:end], color="#333333", linewidth=0.6)
    for s, c in zip(real_i, real_c):
        if start <= s < end:
            for ax in axes[:2]:
                ax.axvline(ms(s), color=COLOURS[c], linewidth=2, alpha=0.6)
    axes[0].set_title("Clean D1 (validation part): real spikes marked by class colour")
    axes[1].set_title(f"Same stretch + level {level} noise: what the detector sees")

    cols = window - PRE                                          # score column of each sample
    ok = (cols >= 0) & (cols < scores.shape[1])
    for k, c in enumerate(COLOURS):
        axes[2].plot(ms(window[ok]), scores[k, cols[ok]], color=COLOURS[c], linewidth=1.2,
                     label=f"Class {c} score")
    axes[2].axhline(threshold, color="#999999", linestyle="--", linewidth=1, label="threshold")
    for s, c in zip(found_i, found_c):
        if start <= s < end:
            axes[2].scatter(ms(s), threshold, marker="v", s=90, color=COLOURS[c],
                            edgecolor="black", zorder=5)
    axes[2].set_ylim(bottom=-3 * abs(threshold))
    axes[2].set_title("Class scores (log-likelihood ratio); ▼ = detection, coloured by the class given")
    axes[2].set_xlabel("Time (ms)")
    axes[2].legend(frameon=False, ncol=6, loc="upper right", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = f"Artifacts/Matched_Filter_Scores_L{level}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
