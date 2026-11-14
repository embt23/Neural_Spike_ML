"""Plot the averaged F curves from fresh_noise_test.py.

Run from the project root:
    python Pos_weight/plot_fresh_noise.py
Saves Artifacts/Pos_Weight/fresh_noise/F_Averaged.png and a zoomed-in
version, F_Averaged_Zoom.png.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

SUMMARY = "Pos_weight/data/fresh_noise/summary.npz"
OUT_DIR = "Artifacts/Pos_Weight/fresh_noise"

# one colour per POS_WEIGHT, in order (colour-blind checked)
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def show_averaged(summary=SUMMARY, below=0.12, above=0.02, xlim=(0, 1)):
    """One panel per level, one averaged F curve per POS_WEIGHT, dot = peak.
    Each panel's y-axis runs from `below` under its lowest peak to `above`
    over its highest peak."""
    s = np.load(summary)
    x = s["thresholds"]
    keys = [k for k in s.files if k.endswith("_mean_f")]
    levels = sorted({int(k.split("_")[0][5:]) for k in keys})
    pws = sorted({int(k.split("_")[1][2:]) for k in keys})

    fig, axes = plt.subplots(len(levels), 1, sharex=True,
                             figsize=(13, 3 * len(levels)), squeeze=False)
    for ax, level in zip(axes[:, 0], levels):
        peaks = []
        for pw, colour in zip(pws, COLOURS):
            mean_f = s[f"level{level}_pw{pw}_mean_f"]
            t, f = s[f"level{level}_pw{pw}_peak"]
            peaks.append(f)
            ax.plot(x, mean_f, color=colour, linewidth=2,
                    label=f"pw{pw}: peak F {f:.3f} at {t:.3f}")
            ax.plot(t, f, "o", color=colour, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.set_ylim(min(peaks) - below, max(peaks) + above)
        ax.set_ylabel(f"Level {level}: mean F")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    axes[-1, 0].set_xlim(*xlim)
    axes[-1, 0].set_xlabel("Threshold (average output of the 9-point group)")
    n = int(s["n_graphs"])
    fig.suptitle(f"Detection F vs threshold, averaged over {n} fresh-noise graphs per level "
                 f"(level 1: 1 graph, no added noise). Run {s['run']}, ±50 samples")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    show_averaged().savefig(f"{OUT_DIR}/F_Averaged.png", dpi=120)
    show_averaged(below=0.03, above=0.005, xlim=(0.15, 0.9)).savefig(
        f"{OUT_DIR}/F_Averaged_Zoom.png", dpi=120)
