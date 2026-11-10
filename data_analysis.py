import numpy as np
import matplotlib.pyplot as plt

FS = 25000  # sampling frequency in Hz

train = np.load("Data/D1_train.npz")
Index = train["Index"]


def spike_spacing(Index, edges=(0, 1, 2, 3, 4, 5, 10, 25, 100, np.inf)):
    """Bar chart of the time (ms) between each spike and the one before it."""
    distance_from_previous = np.diff(Index) / FS * 1000  # samples -> ms
    # DFP_order = Distance From Previous, ordered smallest to largest
    DFP_order = np.sort(distance_from_previous)

    counts, _ = np.histogram(DFP_order, bins=edges)
    labels = [f"{lo:g}–{hi:g}" if hi != np.inf else f"{lo:g}+"
              for lo, hi in zip(edges[:-1], edges[1:])]
    percent = counts / counts.sum() * 100

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, counts)
    plt.bar_label(bars, labels=[f"{c}\n({p:.1f}%)" for c, p in zip(counts, percent)])
    plt.xlabel("Time since previous spike (ms)")
    plt.ylabel("Number of spikes")
    plt.title(f"Spacing between spikes, training data ({len(DFP_order)} gaps)")
    plt.margins(y=0.15)  # room above the bars for the labels
    plt.tight_layout()
    plt.show()

    return distance_from_previous, DFP_order

distance_from_previous, DFP_order = spike_spacing(Index)
