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


def show_spike(i, width=100, before=50):
    """Plot spike number i: `before` samples before its start, `width` after."""
    start = Index[i]
    lo = max(start - before, 0)      # don't run off the start of d
    hi = min(start + width, len(d))  # don't run off the end of d

    t_ms = (np.arange(lo, hi) - start) / FS * 1000  # samples -> ms, 0 = spike start

    plt.figure()
    plt.plot(t_ms, d[lo:hi])
    plt.axvline(0, color="grey", linestyle="--")  # mark the spike start
    plt.xlabel("Time from spike start (ms)")
    plt.ylabel("Amplitude")
    plt.title(f"Spike {i} (class {Class[i]})")


if __name__ == "__main__":
    show_chart(1000)

    for i in range(10):
        show_spike(i)

    plt.show()  # opens every figure at once
