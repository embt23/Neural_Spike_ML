"""Build the classifier's data from what each level's detector finds.

For each level: add fresh noise to D1_train, run that level's detector and
peak picking at its threshold, then match every detection to the real spikes
the way the grader does (one-to-one, closest first, within ±50 samples).
  label 1-5 = the class of the matched spike
  label 0   = fail (no spike nearby, or a second detection on a spike that
              is already taken)
Each detection's window is the noisy signal from 9 samples before the
detected point to 70 after it (80 samples).

Training signals use the detector's training part (D1_train[:SPLIT]),
validation signals the rest. Missed spikes don't appear (nothing to classify).

Run from the project root (levels optional, default 4 5 6):
    python Classifier/make_dataset.py 4 5 6

Saves Classifier/data/level<L>.npz with, for train and val:
  x_*      (n, 80) windows          y_*       label 0-5
  pos_*    detected sample           real_*    matched real spike (-1 if fail)
  signal_* which noisy signal it came from
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "Big_model")
sys.path.insert(0, "Pos_weight")
sys.path.insert(0, "Modular")
import train_big                                    # noqa: E402
import Test_POS_WEIGHT as small                     # noqa: E402
from generate_noise import generate_noise           # noqa: E402
from peak_picking import pick_peaks                 # noqa: E402
from f_score import match                           # noqa: E402

SPLIT = small.SPLIT
BEFORE, AFTER = 9, 70        # window: 9 before the detected point, the point, 70 after
TOLERANCE = 50
OUT_DIR = "Classifier/data"

# level: (model builder, weights, threshold, train signals, val signals)
# thresholds: Big_model/threshold_levels.py (level 6) and
#             Pos_weight/data/fresh_noise/summary.npz (levels 4, 5), 20-signal averages
DETECTORS = {
    4: (small.make_model, "Pos_weight/data/1-20/level4_pw5.pt", 0.620, 10, 5),
    5: (small.make_model, "Pos_weight/data/1-20/level5_pw5.pt", 0.569, 15, 5),
    6: (train_big.make_model, "Big_model/best.pt", 0.600, 20, 5),
}


def detect_and_label(model, threshold, noisy, index, cls):
    """Detections on one noisy signal, with their windows and labels."""
    guesses = pick_peaks(small.predict(model, small.windows(noisy)), threshold=threshold)
    # keep only detections whose whole window fits inside the signal
    guesses = guesses[(guesses >= BEFORE) & (guesses + AFTER < len(noisy))]

    labels = np.zeros(len(guesses), dtype=np.int64)        # 0 = fail
    real = np.full(len(guesses), -1, dtype=np.int64)
    r, g = match(index, guesses, TOLERANCE)
    labels[g], real[g] = cls[r], index[r]

    x = np.stack([noisy[p - BEFORE:p + AFTER + 1] for p in guesses]).astype(np.float32)
    hits = len(g)
    f = 2 * hits / (len(guesses) + len(index)) if hits else 0.0   # = 2PR/(P+R)
    return x, labels, guesses, real, f


def build(level, n_signals, part, model, threshold, d, index, cls):
    xs, ys, ps, rs, ss = [], [], [], [], []
    for s in range(n_signals):
        t0 = time.time()
        noise = generate_noise(level, len(d))
        noisy = (d + noise)[:SPLIT] if part == "train" else (d + noise)[SPLIT:]
        x, y, p, r, f = detect_and_label(model, threshold, noisy, index, cls)
        xs.append(x), ys.append(y), ps.append(p), rs.append(r), ss.append(np.full(len(y), s))
        print(f"  level {level} {part} signal {s + 1}/{n_signals}: {len(y)} detections, "
              f"{(y == 0).sum()} fail, detection F {f:.3f}  ({time.time() - t0:.1f} s)",
              flush=True)
    return {f"x_{part}": np.concatenate(xs), f"y_{part}": np.concatenate(ys),
            f"pos_{part}": np.concatenate(ps), f"real_{part}": np.concatenate(rs),
            f"signal_{part}": np.concatenate(ss)}


def main():
    levels = [int(a) for a in sys.argv[1:]] or [4, 5, 6]
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load("Data/D1_train.npz")
    d, index, cls = data["d"], data["Index"], data["Class"]
    train = index < SPLIT

    for level in levels:
        make_model, weights, threshold, n_train, n_val = DETECTORS[level]
        model = make_model()
        model.load_state_dict(torch.load(weights))
        model.eval()
        print(f"Level {level}: {weights}, threshold {threshold}", flush=True)

        out = build(level, n_train, "train", model, threshold, d,
                    index[train], cls[train])
        out.update(build(level, n_val, "val", model, threshold, d,
                         index[~train] - SPLIT, cls[~train]))
        np.savez_compressed(f"{OUT_DIR}/level{level}.npz", level=level,
                            detector=weights, threshold=threshold,
                            before=BEFORE, after=AFTER, **out)

        for part in ("train", "val"):
            counts = np.bincount(out[f"y_{part}"], minlength=6)
            print(f"Level {level} {part}: {counts.sum()} detections  "
                  f"fail {counts[0]}  " +
                  "  ".join(f"c{c} {counts[c]}" for c in range(1, 6)), flush=True)


if __name__ == "__main__":
    main()
