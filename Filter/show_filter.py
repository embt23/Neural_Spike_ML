"""Picture of what the band-pass filter does to a level 6 signal.

Top: D1 + level 6 noise. Bottom: the same signal after band-pass filtering.
The shaded bands mark where class 3 spikes start (the smallest spikes).

Run from the project root:
    python Filter/show_filter.py
Saves Artifacts/Filter_Before_After.png.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "Modular")
from generate_noise import generate_noise  # noqa: E402
from filtering import bandpass             # noqa: E402

FS = 25000
SHOW = 1500        # samples to show (60 ms)

data = np.load("Data/D1_train.npz")
d, index, cls = data["d"], data["Index"], data["Class"]
noisy = d + generate_noise(6, len(d))
filtered = bandpass(noisy)

# centre the view on a class 3 spike
start = int(index[cls == 3][10]) - SHOW // 2
t_ms = np.arange(SHOW) / FS * 1000
in_view = (index >= start) & (index < start + SHOW)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
for ax, sig, name in zip(axes, (noisy, filtered), ("Raw (level 6)", "Band-pass 300-3000 Hz")):
    ax.plot(t_ms, sig[start:start + SHOW], color="#2a78d6", linewidth=0.8)
    for i, c in zip(index[in_view], cls[in_view]):
        x = (i - start) / FS * 1000
        ax.axvspan(x, x + 2, color="#eb6834" if c == 3 else "#999999", alpha=0.25, linewidth=0)
    ax.set_ylabel(name)
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xlabel("Time (ms)")
fig.suptitle("Band-pass filtering a level 6 signal (orange = class 3 spike, grey = other spikes)")
fig.tight_layout()
plt.savefig("Artifacts/Filter_Before_After.png", dpi=120)
plt.show()
