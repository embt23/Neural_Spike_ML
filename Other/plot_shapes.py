"""Compare the three hidden-layer shapes, for the denoiser and the detector.

Run from the project root (works while the runs are still going):
    python Other/plot_shapes.py
Saves Artifacts/Denoise_Shapes.png and Artifacts/Detect_Shapes_F.png,
and prints a summary table for each.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

SHAPES = ["100-100-25", "300", "70-70-70-70"]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]     # same order as Other/find_threshold.py
GREY = "#8c8c8c"
SMOOTH = 10                                       # checks in the rolling average


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path)]


def rolling(y, n=SMOOTH):
    """Average of the last n values at each point (fewer at the start)."""
    return np.array([y[max(0, i - n + 1):i + 1].mean() for i in range(len(y))])


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def show_denoise():
    base = json.load(open("Denoise/data/baselines.json"))["constant (D1 mean)"]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    print("\nDenoiser (level 6, validation RMS, lower is better)")
    print("  shape          weights  epochs  s/epoch  best RMS  near spike  away")
    for shape, colour in zip(SHAPES, COLOURS):
        rows = load(f"Denoise/data/{shape}/log.jsonl")
        if not rows:
            continue
        e = np.array([r["epoch"] for r in rows])
        for ax, key in zip(axes, ["val_rms", "val_rms_near_spike"]):
            y = np.array([r[key] for r in rows])
            ax.plot(e, y, color=colour, linewidth=2, label=f"61-{shape}-1")
        b = min(rows, key=lambda r: r["val_rms"])
        print(f"  {shape:13s} {rows[0]['n_params']:8,d}  {rows[-1]['epoch']:6d}  "
              f"{rows[-1]['minutes'] * 60 / rows[-1]['epoch']:7.1f}  {b['val_rms']:8.3f}  "
              f"{b['val_rms_near_spike']:10.3f}  {b['val_rms_away']:.3f}")
    for ax, key, name in zip(axes, ["all", "near_spike"], ["All samples", "Near spikes"]):
        ax.axhline(base[key], color=GREY, linestyle="--", linewidth=1.5,
                   label=f"constant guess (D1 mean): {base[key]:.3f}")
        style(ax, f"{name}: validation RMS")
        ax.legend(frameon=False, loc="center right")
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Denoiser shapes on level 6 noise (lower is better; doing nothing = 3.71)")
    fig.tight_layout()
    fig.savefig("Artifacts/Denoise_Shapes.png", dpi=120)


def show_detect():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    overnight = load("Big_model/data/log.jsonl")
    print("\nDetector (level 6, best F on validation, higher is better)")
    print("  shape          weights  epochs  s/epoch  best F  rolling best F")
    last_epoch = 0
    for shape, colour in zip(SHAPES, COLOURS):
        rows = load(f"Detect_shapes/data/{shape}/log.jsonl")
        if not rows:
            continue
        e = np.array([r["epoch"] for r in rows])
        f = np.array([r["f"] for r in rows])
        last_epoch = max(last_epoch, e[-1])
        ax.plot(e, f, color=colour, linewidth=0.8, alpha=0.35)
        ax.plot(e, rolling(f), color=colour, linewidth=2, label=f"50-{shape}-1")
        print(f"  {shape:13s} {rows[0]['n_params']:8,d}  {e[-1]:6d}  "
              f"{rows[-1]['minutes'] * 60 / e[-1]:7.1f}  {f.max():6.3f}  {rolling(f).max():.3f}")
    if overnight and last_epoch:
        rows = [r for r in overnight if r["epoch"] <= last_epoch]
        e = np.array([r["epoch"] for r in rows])
        f = np.array([r["f"] for r in rows])
        ax.plot(e, rolling(f), color=GREY, linewidth=2, linestyle="--",
                label="overnight 50-100-50-25-1")
        print(f"  overnight (same epochs)       {e[-1]:6d}           {f.max():6.3f}  "
              f"{rolling(f).max():.3f}")
    style(ax, "Best F (validation, level 6)")
    ax.set_xlabel("Epoch")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Detector shapes on level 6 noise (thick = rolling average of {SMOOTH} checks)")
    fig.tight_layout()
    fig.savefig("Artifacts/Detect_Shapes_F.png", dpi=120)


if __name__ == "__main__":
    show_denoise()
    show_detect()
    plt.show()
