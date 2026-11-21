import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import sys
sys.path.insert(0, "Modular")
from generate_noise import _psd  # noqa: E402  (the same spectrum measurement the generator uses)

# One distinct colour per level (blue, orange, aqua, yellow, pink, green),
# chosen to stay distinguishable for colour-blind readers
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def load_level(n):
    """d for dataset n. D1 uses the training part only."""
    if n == 1:
        return np.load("Data/D1_train.npz")["d"]
    return loadmat(f"Data/D{n}.mat")["d"].ravel()  # (1, N) -> (N,)


def show_spectra():
    """Plot the power spectral density of D1 to D6 on one log-log chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for n, colour in zip(range(1, 7), COLOURS):
        f, power = _psd(load_level(n))
        ax.plot(f, power, color=colour, linewidth=1.5, label=f"D{n}")
        ax.annotate(f"D{n}", (f[-1], power[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color="#333333")

    ax.set_xscale("log")  # frequencies span 0.01 Hz to 12,500 Hz
    ax.set_yscale("log")  # power spans many orders of magnitude
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (amplitude² per Hz)")
    ax.set_title("Power spectral density of D1 to D6")
    ax.grid(True, which="major", color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()


if __name__ == "__main__":
    show_spectra()
    plt.savefig("Artifacts/PSD_Levels.png")
    plt.show()
