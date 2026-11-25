"""Fourier analysis of each class's average spike (training data only).

The average spike (see composite_spikes.py) is cut to 0 to 4 ms after the
spike start (100 samples). Its baseline (the mean of the 2 ms before the
start) is subtracted, so the flat offset doesn't show up as a 0 Hz spike in
the spectrum. Then the FFT gives how much of each frequency the shape contains.

100 samples at 25 kHz only gives one point every 250 Hz, so the window is
zero-padded to N_FFT samples. That draws a smoother curve between those points;
it doesn't add any new detail.

Run from the project root:
    python Other/spike_spectrum.py
Saves Artifacts/Composate{based on class}/Class<c>/Spectrum.png for each class,
plus All_Classes_Spectrum.png in the parent folder.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "Other")
sys.path.insert(0, "Modular")
from composite_spikes import spike_windows, BEFORE, AFTER, FS, OUT_DIR, COLOURS   # noqa: E402
from generate_noise import generate_noise                                    # noqa: E402

START_MS, END_MS = 0, 4          # part of the spike to analyse
N_FFT = 2000                     # zero-padded length (12.5 Hz between points)
SHOW_HZ = 5000                   # x-axis limit of the plots
N_NOISE = 2000                   # noise windows averaged per level
NOISE_SEED = 4000                # fixed, so the chart is the same every run
GAP_BEFORE, GAP_AFTER = 50, 120  # samples near a spike that don't count as noise


def spectrum(mean):
    """Frequencies (Hz) and amplitude of the FFT of the 0-4 ms part of `mean`."""
    baseline = mean[:BEFORE].mean()
    a = BEFORE + int(START_MS * FS / 1000)
    b = BEFORE + int(END_MS * FS / 1000)
    x = mean[a:b] - baseline
    freqs = np.fft.rfftfreq(N_FFT, d=1 / FS)
    return freqs, np.abs(np.fft.rfft(x, n=N_FFT))


def energy_below(freqs, amp, fraction):
    """Frequency below which `fraction` of the spike's energy lies."""
    energy = np.cumsum(amp**2)
    return freqs[np.searchsorted(energy, fraction * energy[-1])]


def noise_starts(index, length, n, seed):
    """n random window starts where the window is pure noise: no spike starts
    within GAP_AFTER before it to GAP_BEFORE after it (same gaps as tune_filters)."""
    rng = np.random.default_rng(seed)
    s = rng.integers(BEFORE, length - AFTER, size=20 * n)    # extra, most get dropped
    lo = np.searchsorted(index, s - BEFORE - GAP_AFTER)      # first spike after the gap starts
    hi = np.searchsorted(index, s + AFTER + GAP_BEFORE)      # first spike past the gap
    return s[lo == hi][:n]                                   # equal = no spike inside


def noise_spectrum(signal, starts):
    """Typical FFT size of the noise in one window: the root mean square of
    spectrum() over many noise windows (noise is random, so the average of the
    magnitudes, not of the raw windows, which would cancel to zero)."""
    amps = np.stack([spectrum(w)[1] for w in spike_windows(signal, starts)])
    return np.sqrt((amps**2).mean(axis=0))


def show_noise_levels(d, index, spectra, ylim):
    """One panel per noise level: the 5 class spectra plus the noise spectrum.
    Level 1 is D1 as it is; levels 2-6 add generated noise (fixed seeds)."""
    starts = noise_starts(index, len(d), N_NOISE, NOISE_SEED)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

    print("\nlevel  frequencies where each class's spike is above the noise")
    for level, ax in zip(range(1, 7), axes.flat):
        signal = d if level == 1 else d + generate_noise(level, len(d), seed=NOISE_SEED + level)
        noise = noise_spectrum(signal, starts)
        above = []
        for c, (freqs, amp) in spectra.items():
            ax.plot(freqs, amp, color=COLOURS[c], linewidth=1.5, label=f"Class {c}")
            f_above = freqs[(amp > noise) & (freqs <= SHOW_HZ)]
            above.append(f"{c}: {f_above.min():.0f}-{f_above.max():.0f}" if len(f_above) else f"{c}: none")
        print(f"  {level}    " + "   ".join(above) + " Hz")

        ax.fill_between(freqs, noise, color="#555555", alpha=0.25, linewidth=0)
        ax.plot(freqs, noise, color="#333333", linewidth=2, linestyle="--", label="Noise")
        style(ax, ylim)
        ax.set_title(f"Level {level}" + (" (D1 as it is)" if level == 1 else ""))
        ax.label_outer()

    axes.flat[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Average spike vs noise, by frequency (0 to 4 ms windows)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Noise_Levels_Spectrum.png"), dpi=120)
    plt.close(fig)


def style(ax, ylim):
    ax.set_xlim(0, SHOW_HZ)
    ax.set_ylim(ylim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (FFT magnitude)")
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    train = np.load("Data/D1_train.npz")
    d, index, cls = train["d"], train["Index"], train["Class"]
    spectra = {c: spectrum(spike_windows(d, index[cls == c]).mean(axis=0)) for c in COLOURS}
    ylim = (0, 1.05 * max(amp.max() for _, amp in spectra.values()))

    print("class  peak freq   50% energy below   90% below   99% below")
    for c, (freqs, amp) in spectra.items():
        peak = freqs[amp.argmax()]
        f50, f90, f99 = (energy_below(freqs, amp, p) for p in (0.5, 0.9, 0.99))
        print(f"  {c}    {peak:6.0f} Hz      {f50:6.0f} Hz       {f90:6.0f} Hz   {f99:6.0f} Hz")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(freqs, amp, color=COLOURS[c], linewidth=2)
        ax.axvline(f90, color="#999999", linestyle="--", linewidth=1)
        ax.annotate(f"90% of energy below {f90:.0f} Hz", (f90, ylim[1] * 0.9),
                    xytext=(6, 0), textcoords="offset points", color="#555555")
        style(ax, ylim)
        ax.set_title(f"Class {c}: spectrum of the average spike (0 to 4 ms)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"Class{c}", "Spectrum.png"), dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for c, (freqs, amp) in spectra.items():
        ax.plot(freqs, amp, color=COLOURS[c], linewidth=2, label=f"Class {c}")
    style(ax, ylim)
    ax.set_title("Spectrum of each class's average spike (0 to 4 ms)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "All_Classes_Spectrum.png"), dpi=120)
    plt.close(fig)
    show_noise_levels(d, index, spectra, ylim)
    print(f"\nSaved Spectrum.png in each Class folder, plus All_Classes_Spectrum.png"
          f" and Noise_Levels_Spectrum.png in {OUT_DIR}")


if __name__ == "__main__":
    main()
