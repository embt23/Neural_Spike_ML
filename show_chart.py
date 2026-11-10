import numpy as np
import matplotlib.pyplot as plt

FS = 25000  # sampling frequency in Hz

train = np.load("Data/D1_train.npz")
d = train["d"]


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
    plt.show()


show_chart(1000)
