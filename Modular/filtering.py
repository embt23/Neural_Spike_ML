"""Filters that clean up a noisy signal before it goes into the networks.

All three keep spikes at the same sample positions (no time shift), so Index
still lines up after filtering.

Usage from another script (run from the project root):
    sys.path.insert(0, "Modular")
    from filtering import bandpass, wiener, wavelet_denoise
    clean = bandpass(signal)                # 300 to 3000 Hz
    clean = wiener(signal, level=6)         # built from the level 6 noise spectrum
    clean = wavelet_denoise(signal)         # db4, 5 levels, universal soft threshold
"""
from functools import lru_cache

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

from generate_noise import _psd, _extra_noise_psd

FS = 25000          # sampling frequency in Hz
SPLIT = 1_000_000   # D1_train[:SPLIT] is training data; only that part builds the Wiener filter


def bandpass(signal, low=300, high=3000, order=3):
    """Keep frequencies between low and high (Hz), remove the rest.

    low:   below this is slow drift and hum, not spike shape.
    high:  above this is fast hiss, faster than any spike changes.
    order: how sharply the filter cuts off at the edges.

    Uses a Butterworth filter run forwards then backwards (sosfiltfilt), so
    nothing gets shifted in time: spikes stay at the same sample positions.
    """
    sos = butter(order, [low, high], btype="band", fs=FS, output="sos")
    # sosfiltfilt hands back a reversed view of the array; torch can't read
    # those, so return a normal copy
    return np.ascontiguousarray(sosfiltfilt(sos, signal))


@lru_cache(maxsize=None)
def _spectra(level):
    """Spectrum of the clean signal (D1 training part) and of the level's noise,
    on the same frequency bands. Cached, so it's only worked out once per level."""
    f, noise = _extra_noise_psd(level)
    fs_, clean = _psd(np.load("Data/D1_train.npz")["d"][:SPLIT])
    return f, np.interp(f, fs_, clean), noise


def wiener(signal, level=6, strength=1.0):
    """Wiener filter: at each frequency, keep the fraction of the signal that is
    spike rather than noise.

    gain(f) = S(f) / (S(f) + strength * N(f))
      S = power spectrum of clean D1 (spikes), N = power spectrum of the noise
      that D<level> adds. Where noise dominates the gain is near 0; where spikes
      dominate it is near 1.
    strength: how hard to push noise down (1 = the textbook Wiener filter,
      above 1 removes more noise but also more spike).

    The gain is real and applied in the frequency domain, so there is no time shift.
    """
    f, clean, noise = _spectra(int(level))
    gain = clean / (clean + strength * noise)
    freqs = np.fft.rfftfreq(len(signal), d=1 / FS)
    return np.fft.irfft(np.fft.rfft(signal) * np.interp(freqs, f, gain), n=len(signal))


def wavelet_denoise(signal, wavelet="db4", levels=5, rule="universal", mode="soft"):
    """Wavelet denoising: split the signal into frequency bands with a wavelet
    transform, shrink the small coefficients (mostly noise) towards 0, and put
    the signal back together. Sharp spikes make a few big coefficients, so they
    survive; noise spreads over many small ones, so it gets removed.

    wavelet: shape of the wavelet (db4, sym4, sym8, ...).
    levels:  how many bands to split into (each halves the frequency).
    rule:    how big the threshold is, worked out separately for each band:
               "universal": sigma * sqrt(2 ln n), removes almost all noise
               "bayes":     sigma^2 / spread of the spike part, gentler
             sigma = noise size in that band, from the median (robust to spikes).
    mode:    "soft" shrinks every coefficient by the threshold, "hard" only
             zeroes the ones below it.

    The slowest band (the approximation) is left alone.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    n = len(signal)
    for j in range(1, len(coeffs)):          # coeffs[0] is the approximation
        c = coeffs[j]
        sigma = np.median(np.abs(c)) / 0.6745
        if rule == "universal":
            t = sigma * np.sqrt(2 * np.log(n))
        elif rule == "bayes":
            spread = np.sqrt(max(np.var(c) - sigma**2, 0))
            t = sigma**2 / spread if spread > 0 else np.abs(c).max()
        else:
            raise ValueError(f"rule must be 'universal' or 'bayes', got {rule}")
        coeffs[j] = pywt.threshold(c, t, mode=mode)
    return pywt.waverec(coeffs, wavelet)[:n]
