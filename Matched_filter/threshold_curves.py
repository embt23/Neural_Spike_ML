"""F vs threshold for the matched filter + peeling, one panel per level.

For each level, N_SIGNALS tune signals (D1_train[SPLIT:] + generated noise,
the same tune seeds as compare.py, plus more) are run through peel() at every
threshold in THRESHOLDS. Each panel shows:
  - F_total for every signal (thin lines): how much the curve depends on luck
  - their average (bold) and a rolling average of it (dashed, SMOOTH points wide)
  - average F_detect and F_class (thin, coloured): the trade-off underneath
  - the threshold compare.py chose (vertical line) and the top of the smoothed
    average (dot)

Validation data only; the test set is not touched.

Run from the project root (levels optional, default 2-6):
    python Matched_filter/threshold_curves.py
Saves Artifacts/Matched_Filter_Threshold_Curves.png and
Matched_filter/data/threshold_curves.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "Matched_filter")
from matched_filter import build_bank, SPLIT              # noqa: E402
from generate_noise import generate_noise                 # noqa: E402
from compare import grade, TUNE_SEED                      # noqa: E402

N_SIGNALS = 10
THRESHOLDS = np.arange(0, 15.01, 0.25)
SMOOTH = 5                       # rolling average width, in grid points (±0.5)
OUT_JSON = "Matched_filter/data/threshold_curves.json"
CHART = "Artifacts/Matched_Filter_Threshold_Curves.png"


def curves(level):
    """F_detect, F_class, F_total for every signal and threshold: (N_SIGNALS, len(THRESHOLDS), 3)."""
    t = np.load("Data/D1_train.npz")
    d = t["d"]
    val = t["Index"] >= SPLIT
    real_i, real_c = t["Index"][val] - SPLIT, t["Class"][val]
    bank = build_bank(level)
    out = np.zeros((N_SIGNALS, len(THRESHOLDS), 3))
    for s in range(N_SIGNALS):
        x = d[SPLIT:] + generate_noise(level, len(d), seed=TUNE_SEED + 10 * level + s)[SPLIT:]
        for k, th in enumerate(THRESHOLDS):
            out[s, k] = grade(real_i, real_c, *bank.peel(x, th))
    return level, out


def rolling(y, width=SMOOTH):
    """Rolling average; near the edges it averages over the points that exist."""
    kernel = np.ones(width)
    return np.convolve(y, kernel, "same") / np.convolve(np.ones_like(y), kernel, "same")


def main():
    levels = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6]
    with open("Matched_filter/data/thresholds.json") as fh:
        chosen = {int(k): v for k, v in json.load(fh).items()}
    with ProcessPoolExecutor(max_workers=len(levels)) as pool:     # one level per CPU core
        results = dict(pool.map(curves, levels))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    summary = {}
    print("level  chosen   F there   smoothed top   F there   range within 0.005 of the top")
    for ax, level in zip(axes.flat, levels):
        r = results[level]
        mean = r.mean(axis=0)                          # (thresholds, 3)
        smooth = rolling(mean[:, 2])
        top = int(np.argmax(smooth))
        near = THRESHOLDS[smooth >= smooth[top] - 0.005]
        f_chosen = np.interp(chosen[level], THRESHOLDS, smooth)
        summary[level] = {"chosen": round(chosen[level], 3), "f_at_chosen": round(float(f_chosen), 4),
                          "smoothed_top": float(THRESHOLDS[top]), "f_at_top": round(float(smooth[top]), 4),
                          "flat_from": float(near.min()), "flat_to": float(near.max()),
                          "mean_f_total": mean[:, 2].round(4).tolist()}
        print(f"  {level}    {chosen[level]:5.2f}    {f_chosen:.3f}       {THRESHOLDS[top]:5.2f}       "
              f"{smooth[top]:.3f}     {near.min():.2f} to {near.max():.2f}")

        for s in range(N_SIGNALS):
            ax.plot(THRESHOLDS, r[s, :, 2], color="#b8c2c8", linewidth=0.8)
        ax.plot(THRESHOLDS, mean[:, 0], color="#2a78d6", linewidth=1.2, label="F_detect (average)")
        ax.plot(THRESHOLDS, mean[:, 1], color="#eb6834", linewidth=1.2, label="F_class (average)")
        ax.plot(THRESHOLDS, mean[:, 2], color="#222222", linewidth=2.2, label="F_total (average)")
        ax.plot(THRESHOLDS, smooth, color="#0e6c78", linewidth=1.6, linestyle="--",
                label=f"F_total, rolling average ({SMOOTH} points)")
        ax.axvspan(near.min(), near.max(), color="#0e6c78", alpha=0.08, linewidth=0)
        ax.axvline(chosen[level], color="#999999", linestyle=":", linewidth=1.5,
                   label="threshold compare.py chose")
        ax.scatter([THRESHOLDS[top]], [smooth[top]], color="#0e6c78", zorder=5, s=30)
        lo = max(0, mean[:, 1].min() - 0.05)
        ax.set_ylim(lo, 1.0)
        ax.set_title(f"Level {level}: top {smooth[top]:.3f} at {THRESHOLDS[top]:.2f}, "
                     f"chosen {chosen[level]:.2f} gives {f_chosen:.3f}")
        ax.grid(color="#e6e6e6", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Threshold (score)")
        ax.set_ylabel("F")
    for ax in axes.flat[len(levels):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#b8c2c8", linewidth=0.8))
    labels.append(f"F_total, each of {N_SIGNALS} signals")
    axes.flat[-1].legend(handles, labels, frameon=False, loc="center left")
    fig.suptitle("Matched filter + peeling: F vs threshold (shaded: within 0.005 of the top)")
    fig.tight_layout()
    fig.savefig(CHART, dpi=120)
    plt.close(fig)

    with open(OUT_JSON, "w") as fh:
        json.dump({"thresholds": THRESHOLDS.tolist(), "levels": summary}, fh)
    print(f"\nSaved {CHART} and {OUT_JSON}")


if __name__ == "__main__":
    main()
