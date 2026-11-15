"""Band-pass filter: keep the frequencies spikes live in, remove the rest.

Usage from another script (run from the project root):
    sys.path.insert(0, "Modular")
    from filtering import bandpass
    clean = bandpass(signal)              # 300 to 3000 Hz
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 25000   # sampling frequency in Hz


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
