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
from composite_spikes import spike_windows, BEFORE, FS, OUT_DIR, COLOURS   # noqa: E402

START_MS, END_MS = 0, 4          # part of the spike to analyse
N_FFT = 2000                     # zero-padded length (12.5 Hz between points)
SHOW_HZ = 5000                   # x-axis limit of the plots


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
    print(f"\nSaved Spectrum.png in each Class folder and All_Classes_Spectrum.png in {OUT_DIR}")


if __name__ == "__main__":
    main()
