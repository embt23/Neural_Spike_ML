"""Sweep the peak-picking threshold from 0 to 1 and see how F, precision and
recall change.

Run from the project root:
    python Other/find_threshold.py
For each run folder in Pos_weight/data/, saves one chart per level in
Artifacts/Pos_Weight/<run>/Threshold_L<level>.png and prints the best
threshold for each POS_WEIGHT.

One POS_WEIGHT across levels (run, pos_weight, then the levels):
    python Other/find_threshold.py 10-300 10 1 2 3
saves Artifacts/Pos_Weight/<run>/Threshold_PW<pos_weight>.png.

F-score only, zoomed in, one panel per level:
    python Other/find_threshold.py zoom 10-300
saves Artifacts/Pos_Weight/<run>/F_Zoom.png.
"""
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "Modular")
from peak_picking import pick_peaks  # noqa: E402
from f_score import f_detect         # noqa: E402

DATA_DIR = "Pos_weight/data"
OUT_DIR = "Artifacts/Pos_Weight"
RUNS = ("10-300", "1-20")
WIDTH = 9          # peak-picking group size
TOLERANCE = 50     # ± samples for a guess to count as a hit
STEP = 0.001       # threshold step

# one colour per POS_WEIGHT, in order (colour-blind checked)
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# one colour per level, the same as Other/noise_spectrum.py (D1 to D6)
LEVEL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def find_threshold(data, step=STEP, width=WIDTH, tolerance=TOLERANCE):
    """data: path to one results .npz file (uses its last saved epoch).

    Picks every peak once (threshold 0), then for each threshold keeps the
    peaks whose score is above it and scores those guesses.
    Returns a dict with the threshold values, F, precision, recall and the best.
    """
    r = np.load(data)
    guesses, scores = pick_peaks(r["outputs"][-1], width=width,
                                 threshold=0, return_scores=True)
    real = r["val_index"]

    thresholds = np.round(np.arange(0, 1 + step / 2, step), 6)
    f, precision, recall = (np.zeros(len(thresholds)) for _ in range(3))
    for k, t in enumerate(thresholds):
        s = f_detect(real, guesses[scores > t], tolerance)
        f[k], precision[k], recall[k] = s["f"], s["precision"], s["recall"]

    best = int(np.argmax(f))
    return {"thresholds": thresholds, "f": f, "precision": precision,
            "recall": recall, "best_threshold": thresholds[best],
            "best_f": f[best], "level": int(r["level"]),
            "pos_weight": int(r["pos_weight"]), "epoch": int(r["epochs"][-1])}


def show_level(results, title, by="pos_weight"):
    """One figure, three panels (F, precision, recall).
    by="pos_weight": one line per POS_WEIGHT. by="level": one line per level."""
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 10))
    for k, res in enumerate(results):
        if by == "level":
            name, colour = f"level {res['level']}", LEVEL_COLOURS[res["level"] - 1]
        else:
            name, colour = f"pw{res['pos_weight']}", COLOURS[k]
        x = res["thresholds"]
        axes[0].plot(x, res["f"], color=colour, linewidth=2,
                     label=f"{name}: best F {res['best_f']:.3f} "
                           f"at {res['best_threshold']:.3f}")
        axes[0].plot(res["best_threshold"], res["best_f"], "o", color=colour,
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        axes[1].plot(x, res["precision"], color=colour, linewidth=2, label=name)
        axes[2].plot(x, res["recall"], color=colour, linewidth=2, label=name)

    for ax, name in zip(axes, ["F-score", "Precision", "Recall"]):
        ax.set_ylabel(name)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower center")
    axes[1].legend(frameon=False, loc="lower right", ncol=5)
    axes[2].set_xlabel("Threshold (average output of the 9-point group)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_pos_weight(run, pw, levels):
    """One POS_WEIGHT, one line per level. Saves and returns the figure."""
    results = [find_threshold(f"{DATA_DIR}/{run}/level{level}_pw{pw}.npz")
               for level in levels]
    fig = show_level(results, f"Run {run}, POS_WEIGHT {pw}: detection vs threshold "
                              f"(epoch {results[0]['epoch']}, ±{TOLERANCE} samples)",
                     by="level")
    fig.savefig(f"{OUT_DIR}/{run}/Threshold_PW{pw}.png", dpi=120)
    return fig


def show_f_zoom(run, xlim=(0.6, 1), ylim=(0.8, 1), levels=None, name="F_Zoom"):
    """F-score only, one panel per level, one line per POS_WEIGHT, zoomed in.
    levels: which levels to show (default: all). Saves <name>.png."""
    results = [find_threshold(f) for f in glob.glob(f"{DATA_DIR}/{run}/*.npz")]
    levels = levels or sorted({res["level"] for res in results})
    fig, axes = plt.subplots(len(levels), 1, sharex=True, sharey=True,
                             figsize=(13, 3.5 * len(levels)), squeeze=False)
    for ax, level in zip(axes[:, 0], levels):
        here = sorted((res for res in results if res["level"] == level),
                      key=lambda res: res["pos_weight"])
        for res, colour in zip(here, COLOURS):
            ax.plot(res["thresholds"], res["f"], color=colour, linewidth=2,
                    label=f"pw{res['pos_weight']}: best F {res['best_f']:.3f} "
                          f"at {res['best_threshold']:.3f}")
            ax.plot(res["best_threshold"], res["best_f"], "o", color=colour,
                    markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"Level {level}: F-score")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    axes[-1, 0].set_xlabel("Threshold (average output of the 9-point group)")
    fig.suptitle(f"Run {run}: F-score vs threshold, zoomed in "
                 f"(epoch {results[0]['epoch']}, ±{TOLERANCE} samples)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))   # leave room for the title
    fig.savefig(f"{OUT_DIR}/{run}/{name}.png", dpi=120)
    return fig


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "zoom":
    show_f_zoom(sys.argv[2])

elif __name__ == "__main__" and len(sys.argv) > 2:
    show_pos_weight(sys.argv[1], int(sys.argv[2]), [int(a) for a in sys.argv[3:]])

elif __name__ == "__main__":
    for run in RUNS:
        files = glob.glob(f"{DATA_DIR}/{run}/*.npz")
        if not files:
            print(f"{run}: no results yet, skipped")
            continue
        os.makedirs(f"{OUT_DIR}/{run}", exist_ok=True)
        results = [find_threshold(f) for f in files]

        for level in sorted({res["level"] for res in results}):
            here = sorted((res for res in results if res["level"] == level),
                          key=lambda res: res["pos_weight"])
            fig = show_level(here, f"Run {run}, level {level}: detection vs threshold "
                                   f"(epoch {here[0]['epoch']}, ±{TOLERANCE} samples)")
            fig.savefig(f"{OUT_DIR}/{run}/Threshold_L{level}.png", dpi=120)
            plt.close(fig)

            print(f"\n{run}  level {level}")
            print("  pos_weight  best threshold  F      precision  recall")
            for res in here:
                b = int(np.argmax(res["f"]))
                print(f"  {res['pos_weight']:>10}  {res['best_threshold']:>14.3f}  "
                      f"{res['f'][b]:.3f}  {res['precision'][b]:>9.3f}  {res['recall'][b]:.3f}")
