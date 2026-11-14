"""Plot the validation signal and the detector's output under it.

Run from the project root:
    python Pos_weight/plot_outputs.py
Plots every results file in each run folder (Pos_weight/data/<run>/) and
saves the images in Artifacts/Pos_Weight/<run>/.
"""
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

FS = 25000                      # samples per second
SPLIT = 1_000_000               # where validation starts in D1_train
RUNS = ("10-300", "1-20")       # run folders inside Pos_weight/data/
DATA_DIR = "Pos_weight/data"
OUT_DIR = "Artifacts/Pos_Weight"

SIGNAL = "#2a78d6"   # blue
OUTPUT = "#eb6834"   # orange
CLEAN = "#333333"    # dark grey
SPIKE = "#bbbbbb"    # light grey


def show_output(path, epoch=100, x=2000):
    """path: one results .npz file.
    Top: noisy validation signal (with clean D1 over it).
    Bottom: detector output at the chosen epoch. Grey lines = real spikes."""
    r = np.load(path)
    level, pw = int(r["level"]), int(r["pos_weight"])
    row = list(r["epochs"]).index(epoch)
    noisy = r["val_d"][:x]
    out = r["outputs"][row, :x].astype(np.float32)
    clean = np.load("Data/D1_train.npz")["d"][SPLIT:SPLIT + x]
    spikes = r["val_index"][r["val_index"] < x]

    t_ms = np.arange(x) / FS * 1000  # sample number -> ms
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 6),
                                      height_ratios=[2, 1])

    label = "val_d (with noise)" if level > 1 else "val_d (no added noise)"
    top.plot(t_ms, noisy, color=SIGNAL, linewidth=0.8, label=label)
    if level > 1:
        top.plot(t_ms, clean, color=CLEAN, linewidth=0.6, label="D1 (clean)")
    top.set_ylabel("Signal")
    top.legend(frameon=False, loc="upper right")

    bottom.plot(t_ms, out, color=OUTPUT, linewidth=1)
    bottom.set_ylim(-0.05, 1.05)
    bottom.set_ylabel(f"Output (epoch {epoch})")
    bottom.set_xlabel("Time (ms)")

    for ax in (top, bottom):
        for s in spikes:  # real spike starts
            ax.axvline(s / FS * 1000, color=SPIKE, linewidth=1, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Level {level}, POS_WEIGHT {pw}: first {x:,} validation samples "
                 f"(grey lines = real spike starts)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for run in RUNS:
        os.makedirs(f"{OUT_DIR}/{run}", exist_ok=True)
        for path in sorted(glob.glob(f"{DATA_DIR}/{run}/*.npz")):
            fig = show_output(path)
            r = np.load(path)
            name = f"Output_L{int(r['level'])}_PW{int(r['pos_weight'])}.png"
            fig.savefig(f"{OUT_DIR}/{run}/{name}", dpi=120)
            plt.close(fig)
