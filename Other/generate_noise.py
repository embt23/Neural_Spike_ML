import time
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import periodogram

FS = 25000          # sampling frequency in Hz
N_BANDS = 150       # number of log-spaced frequency bands for the spectrum
LEVELS = (2, 3, 4, 5, 6)


def _psd(x):
    """Power spectral density of x: how much power sits at each frequency.

    Uses the whole signal (so very slow drift is captured), then averages
    within log-spaced frequency bands to smooth out the randomness."""
    f, p = periodogram(x, fs=FS)
    edges = np.geomspace(0.01, FS / 2, N_BANDS + 1)
    band = np.digitize(f, edges) - 1           # which band each frequency is in
    keep = (band >= 0) & (band < N_BANDS)
    total = np.bincount(band[keep], p[keep], N_BANDS)
    count = np.bincount(band[keep], minlength=N_BANDS)
    centres = np.sqrt(edges[:-1] * edges[1:])  # middle of each band (log scale)
    full = count > 0                           # lowest bands can be empty
    return centres[full], total[full] / count[full]


@lru_cache(maxsize=None)
def _extra_noise_psd(level):
    """Spectrum of the noise that D<level> has on top of D1.

    PSD(D<level>) - PSD(D1): spikes appear in both, so they mostly cancel,
    and what's left is the extra noise. Only D1's training part is used.
    Cached, so each level's .mat file is only read once per run."""
    d1 = np.load("Data/D1_train.npz")["d"]
    dn = loadmat(f"Data/D{level}.mat")["d"].ravel()  # (1, N) -> (N,)

    f1, p1 = _psd(d1)
    f, pn = _psd(dn)
    p1 = np.interp(f, f1, p1)  # line up with D<level>'s bands (lengths differ)

    extra = np.clip(pn - p1, 0, None)  # power can't be negative
    return f, extra


def generate_noise(level, n_samples=1_200_000):
    """Return a brand-new noise signal shaped like D<level>'s extra noise.

    level: 2 to 6 (or "D2" to "D6"). The seed comes from the clock, so
    every call gives different noise. Add the result to D1's d."""
    level = int(str(level).upper().lstrip("D"))
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level}")

    seed = time.time_ns()
    rng = np.random.default_rng(seed)

    # 1. white noise: equal power at every frequency
    white = rng.standard_normal(n_samples)

    # 2. reshape its spectrum to match the target level
    f, extra = _extra_noise_psd(level)
    freqs = np.fft.rfftfreq(n_samples, d=1 / FS)
    target = np.interp(freqs, f, extra)                 # target PSD at each FFT frequency
    target[0] = 0                                       # no constant offset
    spectrum = np.fft.rfft(white) * np.sqrt(target * FS / 2)  # white noise PSD is 2/FS

    return np.fft.irfft(spectrum, n=n_samples)


def show_noise_levels(x=2000):
    """Plot D1 and D1 + generated noise for each level, stacked on shared
    axes, laid out like Artifacts/D_Level.png so the two can be compared."""
    d1 = np.load("Data/D1_train.npz")["d"]
    t_ms = np.arange(x) / FS * 1000  # sample number -> ms

    fig, axes = plt.subplots(6, 1, sharex=True, sharey=True, figsize=(10, 10))
    axes[0].plot(t_ms, d1[:x], linewidth=0.8)
    axes[0].set_ylabel("D1")

    for ax, level in zip(axes[1:], LEVELS):
        noisy = d1 + generate_noise(level, len(d1))
        ax.plot(t_ms, noisy[:x], linewidth=0.8)
        ax.set_ylabel(f"D1 + noise {level}")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"First {x:,} samples of D1 with generated noise")
    fig.tight_layout()


if __name__ == "__main__":
    show_noise_levels()
    plt.savefig("Artifacts/D_Level_Generated.png")
    plt.show()
