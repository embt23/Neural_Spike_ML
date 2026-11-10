import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

FS = 25000 # sampling Frequency

# load in the data
D1 = loadmat("Data/D1.mat")
d = D1["d"].ravel()
Index = D1["Index"].ravel()
Class = D1["Class"].ravel()

# show spike by Index #
def show_spike(i, width=100, before=10):
    start = Index[i] - 1             # MATLAB counts from 1, Python from 0
    lo = max(start - before, 0)      # don't run off the start of d
    hi = min(start + width, len(d))  # don't run off the end of d

    t_ms = (np.arange(lo, hi) - start) / FS * 1000  # samples -> ms, 0 = spike start

    plt.plot(t_ms, d[lo:hi])
    plt.axvline(0, color="grey", linestyle="--")  # mark the spike start
    plt.xlabel("Time from spike start (ms)")
    plt.ylabel("Amplitude")
    plt.title(f"Spike {i} (class {Class[i]})")
    plt.show()

show_spike(1)
show_spike(2)
show_spike(3)
show_spike(4)
show_spike(5)
