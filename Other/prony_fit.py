"""Prony's method: describe each class's average spike with a few poles.

Idea: a spike looks like the response of a simple system that got one kick.
Such a response is a sum of decaying (and maybe oscillating) exponentials:

    x[n] = sum over k of  h_k * z_k ** n

Each z_k is a pole (the z-transform version of a Laplace pole). From a pole:
    decay time = how long that part takes to shrink to 37% (1/e) of its size
    frequency  = how fast it oscillates (0 Hz = a plain exponential, no wobble)
Complex poles come in mirror pairs (same decay and frequency); together they
make one real decaying sine wave.

Prony's method, in two least-squares steps:
  1. Linear prediction: find a_1..a_p so each sample is predicted from the p
     before it: x[n] = -(a_1 x[n-1] + ... + a_p x[n-p]). The roots of
     z^p + a_1 z^(p-1) + ... + a_p are the poles.
  2. With the poles fixed, find the sizes h_k that fit x best.

The average spike (training data only) is cut to START_MS to END_MS (0 to 8 ms) after the spike
start, with the baseline (mean of the 2 ms before) subtracted, as in
spike_spectrum.py. Every order from 1 to MAX_ORDER is tried; the chosen order
is the last one before adding a pole stops helping (adds less than MIN_GAIN
of the spike's energy). Past that point the extra poles only fit noise.

Run from the project root:
    python Other/prony_fit.py
Saves Prony_Order.png, Prony_Fit.png, Prony_S_Plane.png and prony.json in
Artifacts/Composate{based on class}/.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "Other")
from composite_spikes import spike_windows, BEFORE, FS, OUT_DIR, COLOURS   # noqa: E402

START_MS, END_MS = 0, 8          # part of the spike to fit
MAX_ORDER = 8                    # most poles tried
MIN_GAIN = 0.001                 # a pole must add at least 0.1% of the energy to be kept


def average_spike(d, index):
    """START_MS to END_MS of the average spike, baseline subtracted."""
    mean = spike_windows(d, index).mean(axis=0)
    a = BEFORE + int(START_MS * FS / 1000)
    b = BEFORE + int(END_MS * FS / 1000)
    return mean[a:b] - mean[:BEFORE].mean()


def prony(x, p):
    """Fit x with p poles. Returns poles z, sizes h and the fitted curve."""
    n = len(x)
    # 1. linear prediction: rows are [x[k-1], ..., x[k-p]] for k = p..n-1
    past = np.column_stack([x[p - j:n - j] for j in range(1, p + 1)])
    a, *_ = np.linalg.lstsq(past, -x[p:], rcond=None)
    z = np.roots(np.concatenate([[1], a]))
    # 2. sizes: column k is z_k ** 0, z_k ** 1, ..., z_k ** (n-1)
    powers = z[np.newaxis, :] ** np.arange(n)[:, np.newaxis]
    h, *_ = np.linalg.lstsq(powers, x.astype(complex), rcond=None)
    return z, h, (powers @ h).real


def explained(x, fit):
    """Share of the spike's energy the fit captures (1 = perfect)."""
    return 1 - np.sum((x - fit) ** 2) / np.sum(x ** 2)


def describe(z, h):
    """Plain-language description of each pole (mirror pairs listed once)."""
    poles = []
    for zk, hk in zip(z, h):
        if zk.imag < -1e-12:                    # the mirror of a pair: skip
            continue
        pair = abs(zk.imag) > 1e-12
        decay_ms = -1000 / (FS * np.log(abs(zk))) if abs(zk) < 1 else float("inf")
        if decay_ms < 1000 / FS:                # gone within one sample (0.04 ms)
            kind = "start correction"           # only reshapes the first sample or two
        else:
            kind = "decaying sine (pole pair)" if pair else "plain exponential"
        poles.append({
            "frequency_hz": round(float(abs(np.angle(zk)) * FS / (2 * np.pi)), 1),
            "decay_ms": round(float(decay_ms), 3),
            "size": round(float(abs(hk) * (2 if pair else 1)), 3),   # pair adds up
            "kind": kind,
            "grows": bool(abs(zk) >= 1),        # a growing pole is not physical
        })
    return sorted(poles, key=lambda q: -q["size"])


def main():
    train = np.load("Data/D1_train.npz")
    d, index, cls = train["d"], train["Index"], train["Class"]
    spikes = {c: average_spike(d, index[cls == c]) for c in COLOURS}
    orders = range(1, MAX_ORDER + 1)

    scores = {c: [explained(x, prony(x, p)[2]) for p in orders] for c, x in spikes.items()}
    chosen = {c: next((p for p in orders[:-1] if scores[c][p] - scores[c][p - 1] < MIN_GAIN),
                      MAX_ORDER)    # scores[c][p] is the score with p + 1 poles
              for c in spikes}

    # --- print and save the chosen model of each class ---
    results, s_poles = {}, {}
    for c, x in spikes.items():
        z, h, fit = prony(x, chosen[c])
        # Laplace poles: s = FS * ln(z). Start corrections (gone within one
        # sample, |z| < 1/e) are left out: they aren't part of the shape.
        s_poles[c] = FS * np.log(z[abs(z) >= np.exp(-1)].astype(complex))
        poles = describe(z, h)
        results[c] = {"order": chosen[c], "explained": round(scores[c][chosen[c] - 1], 4),
                      "explained_by_order": [round(s, 4) for s in scores[c]], "poles": poles}
        print(f"\nClass {c}: {chosen[c]} poles explain {100 * results[c]['explained']:.1f}% of the shape")
        for q in poles:
            note = "  (GROWS: not physical)" if q["grows"] else ""
            print(f"    {q['kind']:26s} {q['frequency_hz']:7.0f} Hz   decay {q['decay_ms']:6.3f} ms"
                  f"   size {q['size']:7.2f}{note}")

    with open(os.path.join(OUT_DIR, "prony.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- chart 1: how much each number of poles explains ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in spikes:
        ax.plot(orders, [100 * s for s in scores[c]], "o-", color=COLOURS[c], label=f"Class {c}")
    ax.set_xlabel("Number of poles")
    ax.set_ylabel("Shape explained (%)")
    ax.set_ylim(min(0, 100 * min(min(s) for s in scores.values())), 101)
    ax.set_title("Prony fit quality vs number of poles")
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Prony_Order.png"), dpi=120)
    plt.close(fig)

    # --- chart 2: average spike vs its Prony fit, one panel per class ---
    t_ms = START_MS + np.arange(len(next(iter(spikes.values())))) / FS * 1000
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for ax, (c, x) in zip(axes, spikes.items()):
        ax.plot(t_ms, x, color=COLOURS[c], linewidth=2.5, label="Average spike")
        ax.plot(t_ms, prony(x, chosen[c])[2], color="#333333", linestyle="--",
                linewidth=1.5, label="Prony fit")
        ax.set_title(f"Class {c}: {chosen[c]} poles")
        ax.set_xlabel("Time from spike start (ms)")
        ax.grid(color="#e6e6e6", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Prony_Fit.png"), dpi=120)
    plt.close(fig)
    # --- chart 3: the s-plane, one dot per pole ---
    # x = sigma (real part): how fast it decays, further left = faster.
    # y = omega (imaginary part): how fast it oscillates. Mirror pairs show
    # as two dots, one above and one below the x-axis.
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, sk in s_poles.items():
        ax.scatter(sk.real, sk.imag, s=70, color=COLOURS[c], zorder=3, label=f"Class {c}")
        top = sk[sk.imag.argmax()]
        offset = (-16, -18) if c == 4 else (8, 6)   # 4 sits right under 5: label it below
        ax.annotate(f"{c}", (top.real, top.imag), xytext=offset,
                    textcoords="offset points", color=COLOURS[c], fontweight="bold")
    ax.axhline(0, color="#999999", linewidth=1)
    ax.axvline(0, color="#999999", linewidth=1)
    left = 1.15 * min(sk.real.min() for sk in s_poles.values())
    ax.set_xlim(left, -0.08 * left)
    ax.set_xlabel("σ, real part (1/s): decay, further left = dies out faster")
    ax.set_ylabel("ω, imaginary part (rad/s): oscillation")
    hz = ax.secondary_yaxis("right", functions=(lambda w: w / (2 * np.pi),
                                                lambda f: f * 2 * np.pi))
    hz.set_ylabel("Frequency (Hz)")
    ax.set_title("Laplace poles of each class's average spike (s-plane)")
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.spines[["top"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Prony_S_Plane.png"), dpi=120)
    plt.close(fig)
    print(f"\nSaved Prony_Order.png, Prony_Fit.png, Prony_S_Plane.png and prony.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
