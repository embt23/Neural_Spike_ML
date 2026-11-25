"""Quick tuning: pick the best of 9 settings for each filter, without training.

For each setting, N_SIGNALS level 6 validation signals are made
(D1_train[SPLIT:] + noise with fixed seeds) and filtered. Then:

  spike height: for each class, average the filtered windows at that class's
      true spikes (9 before to 70 after, like the classifier sees). Averaging
      cancels the noise, leaving what the filter did to the spike. Height =
      max - min of that average.
  noise size:   standard deviation of the filtered signal well away from
      any spike (more than GAP_BEFORE before or GAP_AFTER after a spike start).
  score:        height / noise size per class (how much spikes stand out),
      averaged over the 5 classes. Higher is better.

The unfiltered signal is scored too, as a reference. Only validation data is
used; the test set is not touched.

Run from the project root:
    python Filter_compare/tune_filters.py
Saves Filter_compare/data/tuning.json (every score + the winner of each filter).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "Modular")
from generate_noise import generate_noise                   # noqa: E402
from filtering import bandpass, wiener, wavelet_denoise     # noqa: E402

LEVEL = 6
SPLIT = 1_000_000          # validation = D1_train[SPLIT:]
N_SIGNALS = 5              # noisy validation signals per setting
TUNE_SEED = 3000           # noise seed = TUNE_SEED + signal number (unused elsewhere)
BEFORE, AFTER = 9, 70      # window around a spike start, same as the classifier
GAP_BEFORE, GAP_AFTER = 50, 120   # samples near a spike that don't count as noise
CLASSES = (1, 2, 3, 4, 5)
OUT_DIR = "Filter_compare/data"

# 9 settings per filter: (name, function that filters a signal)
SETTINGS = {
    "none": [("unfiltered", lambda x: x)],
    "bandpass": [(f"low {lo}, high {hi}, order 3",
                  lambda x, lo=lo, hi=hi: bandpass(x, lo, hi, order=3))
                 for lo in (100, 300, 500) for hi in (2000, 3000, 5000)],
    "wiener": [(f"strength {s:.2f}", lambda x, s=s: wiener(x, LEVEL, strength=s))
               for s in np.geomspace(0.25, 4, 9)],
    "wavelet": [(f"{w}, 5 levels, {rule} {mode}",
                 lambda x, w=w, rule=rule, mode=mode: wavelet_denoise(x, w, 5, rule, mode))
                for w in ("db4", "sym4", "sym8")
                for rule, mode in (("universal", "soft"), ("universal", "hard"), ("bayes", "soft"))],
}


def noise_mask(index, n):
    """True at samples far enough from every spike to count as pure noise."""
    near = np.zeros(n, dtype=bool)
    for i in index:
        near[max(i - GAP_BEFORE, 0):i + GAP_AFTER] = True
    return ~near


def score(filtered, index, cls, mask):
    """Spike height / noise size for each class."""
    noise = filtered[mask].std()
    out = []
    for c in CLASSES:
        starts = index[cls == c]
        template = np.mean([filtered[i - BEFORE:i + AFTER + 1] for i in starts], axis=0)
        out.append((template.max() - template.min()) / noise)
    return np.array(out)


def main():
    data = np.load("Data/D1_train.npz")
    d, index, cls = data["d"], data["Index"], data["Class"]
    clean = d[SPLIT:]
    keep = (index >= SPLIT + BEFORE) & (index + AFTER < len(d))
    index, cls = index[keep] - SPLIT, cls[keep]
    mask = noise_mask(index, len(clean))
    print(f"Level {LEVEL}, {len(index)} validation spikes, {mask.mean():.0%} of samples "
          f"count as noise, {N_SIGNALS} noise signals per setting\n")

    noisy = [clean + generate_noise(LEVEL, len(d), seed=TUNE_SEED + s)[SPLIT:]
             for s in range(N_SIGNALS)]

    results = {}
    for family, settings in SETTINGS.items():
        print(f"{family}" + " " * 32 + "class 1-5 SNR             mean")
        rows = []
        for name, f in settings:
            snr = np.mean([score(f(x), index, cls, mask) for x in noisy], axis=0)
            rows.append({"setting": name, "snr": snr.round(3).tolist(),
                         "score": round(float(snr.mean()), 3)})
            print(f"  {name:34s}" + " ".join(f"{v:5.2f}" for v in snr) + f"   {snr.mean():5.2f}",
                  flush=True)
        best = max(rows, key=lambda r: r["score"])
        results[family] = {"settings": rows, "best": best}
        print(f"  -> best: {best['setting']} ({best['score']:.2f})\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/tuning.json", "w") as fh:
        json.dump({"level": LEVEL, "n_signals": N_SIGNALS, "tune_seed": TUNE_SEED,
                   "results": results}, fh, indent=1)
    print(f"Saved {OUT_DIR}/tuning.json")


if __name__ == "__main__":
    main()
