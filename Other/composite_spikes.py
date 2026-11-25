"""Average spike shape of each class, from the training data only.

Every spike of a class is cut out (BEFORE samples before its start, AFTER
after) and the windows are averaged sample by sample. The noise is different
in every window, so it mostly cancels; the spike shape is the same, so it stays.

Each image shows:
  - every individual spike of the class, faint
  - the band of ±1 standard deviation around the average (how much spikes vary)
  - the average spike, bold

Run from the project root:
    python Other/composite_spikes.py
Saves Artifacts/Composate{based on class}/Class<c>/Composite.png for each class,
plus All_Classes.png (the five averages on one chart) in the parent folder.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

FS = 25000                      # sampling frequency in Hz
BEFORE, AFTER = 50, 100         # same window as Other/visualize.py show_spike
OUT_DIR = "Artifacts/Composate{based on class}"
COLOURS = {1: "#257de0", 2: "#f47230", 3: "#00b894", 4: "#f2a600", 5: "#ee7fac"}


def spike_windows(d, index):
    """One row per spike: d from BEFORE samples before its start to AFTER after.
    Spikes too close to either end of d are skipped."""
    index = index[(index >= BEFORE) & (index + AFTER <= len(d))]
    return np.stack([d[i - BEFORE:i + AFTER] for i in index])


def style(ax, ylim):
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)   # spike start
    ax.set_ylim(ylim)
    ax.set_xlabel("Time from spike start (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    train = np.load("Data/D1_train.npz")
    d, index, cls = train["d"], train["Index"], train["Class"]
    t_ms = np.arange(-BEFORE, AFTER) / FS * 1000

    windows = {c: spike_windows(d, index[cls == c]) for c in COLOURS}
    lo = min(w.min() for w in windows.values())
    hi = max(w.max() for w in windows.values())
    pad = 0.05 * (hi - lo)
    ylim = (lo - pad, hi + pad)          # same y-axis on every image

    for c, w in windows.items():
        mean, std = w.mean(axis=0), w.std(axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_ms, w.T, color=COLOURS[c], alpha=0.03, linewidth=0.8)
        ax.fill_between(t_ms, mean - std, mean + std, color=COLOURS[c], alpha=0.25,
                        linewidth=0, label="±1 standard deviation")
        ax.plot(t_ms, mean, color="#1a1a1a", linewidth=2, label="Average")
        style(ax, ylim)
        ax.set_title(f"Class {c}: average of {len(w)} training spikes")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()

        folder = os.path.join(OUT_DIR, f"Class{c}")
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, "Composite.png"), dpi=120)
        plt.close(fig)
        print(f"Class {c}: {len(w)} spikes, peak of average {mean.max():.2f} "
              f"at {t_ms[mean.argmax()]:.2f} ms -> {folder}/Composite.png")

    # the five averages on one chart, to compare shapes
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, w in windows.items():
        ax.plot(t_ms, w.mean(axis=0), color=COLOURS[c], linewidth=2, label=f"Class {c}")
    style(ax, ylim)
    ax.set_title("Average spike of each class (training data)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "All_Classes.png"), dpi=120)
    plt.close(fig)
    print(f"All classes -> {OUT_DIR}/All_Classes.png")


if __name__ == "__main__":
    main()
