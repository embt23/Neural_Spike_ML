import os

import numpy as np
import matplotlib.pyplot as plt

FS = 25000  # sampling frequency in Hz

# load in the training data
train = np.load("Data/D1_train.npz")
d = train["d"]
Index = train["Index"]
Class = train["Class"]


def show_chart(x):
    """Plot the first x samples of the training recording."""
    x = min(x, len(d))       # can't show more samples than exist
    t = np.arange(x) / FS    # sample number -> seconds

    if t[-1] < 1:            # under 1 second: show milliseconds
        t, unit = t * 1000, "ms"
    else:
        unit = "s"

    plt.figure()
    plt.plot(t, d[:x], linewidth=0.8)
    plt.xlabel(f"Time ({unit})")
    plt.ylabel("Amplitude")
    plt.title(f"First {x:,} samples of training data ({x / FS:.3g} s)")


def show_spike(i, width=100, before=50, ylim=None):
    """Plot spike number i: `before` samples before its start, `width` after.
    ylim=(low, high) fixes the y-axis; None lets matplotlib choose."""
    start = Index[i]
    lo = max(start - before, 0)      # don't run off the start of d
    hi = min(start + width, len(d))  # don't run off the end of d

    t_ms = (np.arange(lo, hi) - start) / FS * 1000  # samples -> ms, 0 = spike start

    plt.figure()
    plt.plot(t_ms, d[lo:hi])
    plt.axvline(0, color="grey", linestyle="--")  # mark the spike start
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Time from spike start (ms)")
    plt.ylabel("Amplitude")
    plt.title(f"Spike {i} (class {Class[i]})")


def save_class_examples(n=20, out_dir="Artifacts/Spikes{based on class}",
                        width=100, before=50):
    """Save the first n spikes (in time order) of each class as PNGs,
    one folder per class, e.g. Spikes{based on class}/Class1/Spike12.png.
    Every image uses the same x and y axes so classes can be compared."""
    # spike numbers to plot, per class
    chosen = {c: np.where(Class == c)[0][:n] for c in np.unique(Class)}

    # Shared y-axis: lowest and highest point over every window we'll plot
    windows = [d[max(Index[i] - before, 0):min(Index[i] + width, len(d))]
               for picks in chosen.values() for i in picks]
    y_min = min(w.min() for w in windows)
    y_max = max(w.max() for w in windows)
    pad = 0.05 * (y_max - y_min)
    ylim = (y_min - pad, y_max + pad)
    print(f"Shared y-axis: {ylim[0]:.2f} to {ylim[1]:.2f}")

    for c, picks in chosen.items():
        folder = os.path.join(out_dir, f"Class{c}")
        os.makedirs(folder, exist_ok=True)

        for i in picks:
            show_spike(i, width=width, before=before, ylim=ylim)
            plt.savefig(os.path.join(folder, f"Spike{i}.png"))
            plt.close()  # don't keep every figure open

        print(f"Class {c}: saved {len(picks)} images to {folder}")


if __name__ == "__main__":
    save_class_examples()

    show_chart(1000)

    for i in range(10):
        show_spike(i)

    plt.show()  # opens every figure at once
