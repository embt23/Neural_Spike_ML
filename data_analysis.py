import os

import numpy as np
import matplotlib.pyplot as plt

FS = 25000  # sampling frequency in Hz
MS = FS // 1000  # samples per millisecond (25)

train = np.load("Data/D1_train.npz")
d = train["d"]
Index = train["Index"]
Class = train["Class"]


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


# Gap groups in samples (1 ms = 25 samples). Each range is [lo, hi).
PAIR_GROUPS = {
    "0-1ms": (0 * MS, 1 * MS),
    "1-2ms": (1 * MS, 2 * MS),
    "2-3ms": (2 * MS, 3 * MS),
    "3-5ms": (3 * MS, 5 * MS),
}


def save_spike_pairs(d, Index, Class, out_dir="Artifacts/Spikes{pairs}",
                     before=1 * MS, after=8 * MS, groups=PAIR_GROUPS):
    """Save a PNG of every close spike pair into a folder per gap group.

    For each spike k, gap = Index[k+1] - Index[k]. If the gap falls in a
    group, plot from `before` samples before spike k to `after` samples
    after it. All images share one y-axis, set by the tallest and lowest
    points across every window that gets plotted.
    """
    gaps = np.diff(Index)  # gaps[k] = distance from spike k to spike k+1

    # 1) Work out which pairs go in which folder
    pairs = []  # (k, folder name)
    for name, (lo, hi) in groups.items():
        for k in np.where((gaps >= lo) & (gaps < hi))[0]:
            start = Index[k]
            if start - before >= 0 and start + after <= len(d):  # stay inside d
                pairs.append((k, name))

    # 2) Shared y-axis: min and max over all windows that will be plotted
    windows = [d[Index[k] - before:Index[k] + after] for k, _ in pairs]
    y_min = min(w.min() for w in windows)
    y_max = max(w.max() for w in windows)
    pad = 0.05 * (y_max - y_min)
    ylim = (y_min - pad, y_max + pad)

    t_ms = np.arange(-before, after) / FS * 1000  # 0 = first spike start

    # 3) One image per pair
    for (k, name), window in zip(pairs, windows):
        gap_ms = gaps[k] / FS * 1000

        plt.figure()
        plt.plot(t_ms, window)
        plt.axvline(0, color="grey", linestyle="--")          # first spike
        plt.axvline(gap_ms, color="red", linestyle="--")      # second spike
        plt.ylim(ylim)
        plt.xlabel("Time from first spike start (ms)")
        plt.ylabel("Amplitude")
        plt.title(f"Spike {k} → {k + 1}, gap {gap_ms:.2f} ms "
                  f"(classes {Class[k]}/{Class[k + 1]})")

        folder = os.path.join(out_dir, name)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f"Spike{k}.png"))
        plt.close()  # free memory, otherwise hundreds of figures stay open

    counts = {name: sum(1 for _, n in pairs if n == name) for name in groups}
    print(f"Saved {len(pairs)} images to {out_dir}: {counts}")
    print(f"Shared y-axis: {ylim[0]:.2f} to {ylim[1]:.2f}")


if __name__ == "__main__":
    distance_from_previous, DFP_order = spike_spacing(Index)
    save_spike_pairs(d, Index, Class)
