import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

FS = 25000 # sampling Frequency

# load in the data
train = np.load("Data/D1_train.npz")
d = train["d"]
Index = train["Index"]
Class = train["Class"]

# show spike by Index #
def show_spike(i, width=100, before=50):
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

for i in range(20):
    show_spike(i)

plt.show()
