"""One detector per class per noise level, everything in one file.

For each class (1-5) and each level (2-6) a separate network learns to find
the spikes of that ONE class in D1 + generated noise of that level.

  Labels:    a window is 1 if it starts within ±4 samples of a spike of the
             chosen class (acceptance window 9), otherwise 0. Windows on
             spikes of the other classes are 0.
  Network:   50-200-100-50-1, ReLU between layers, sigmoid on the output.
  Loss:      BCEWithLogitsLoss, POS_WEIGHT 25. Adam, lr 0.001, batch 256.
  Training:  D1_train[:SPLIT] + brand-new noise every epoch.
  Checking:  every CHECK_EVERY epochs, on D1_train[SPLIT:] + one fixed noise
             signal per level (seed VAL_SEED + level). The output for every
             window goes through peak picking (group of 9), then F against
             the chosen class's spikes (±50 samples) at every threshold from
             0 to 1. The best F is logged; the best model is saved.

Run from the project root: hours per run, then the classes (default 1-5),
optionally --levels (default 2-6). Each run uses THREADS CPU cores, so two
terminals can run side by side:
    python 1detector-per-class/main.py 1 1 2 3
    python 1detector-per-class/main.py 1 4 5

Saves to 1detector-per-class/data/class<C>_level<L>/:
  log.jsonl   one line per check: epoch, minutes, loss, best threshold, F, P, R
  best.pt     the model with the best validation F so far
  last.pt     the model at the end
"""
import argparse
import json
import os
import time
from functools import lru_cache

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import periodogram
from torch import nn

# ---- settings -------------------------------------------------------------
SHAPE = [50, 200, 100, 50, 1]   # first number = input window length (2 ms)
WINDOW = 9                 # acceptance window: ±4 samples, also the peak-picking group
POS_WEIGHT = 25
BATCH = 256
LR = 0.001
CHECK_EVERY = 25           # epochs between validation checks
SWEEP_STEP = 0.01          # threshold step when checking
TOLERANCE = 50             # ± samples for a guess to count as a hit (grader rule)
COLLAPSE_EPOCH = 100       # warn if F is still below COLLAPSE_F at this epoch
COLLAPSE_F = 0.1
THREADS = 4                # CPU cores per run
SPLIT = 1_000_000          # train = D1_train[:SPLIT], validation = the rest
SEED = 0
VAL_SEED = 1000            # validation noise seed = VAL_SEED + level
FS = 25000                 # sampling frequency in Hz
N_BANDS = 150              # log-spaced frequency bands for the noise spectrum
LEVELS = (2, 3, 4, 5, 6)
CLASSES = (1, 2, 3, 4, 5)
OUT_DIR = "1detector-per-class/data"


# ---- noise ----------------------------------------------------------------
def _psd(x):
    """Power spectral density of x, averaged within log-spaced frequency bands."""
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
    """Spectrum of the noise D<level> has on top of D1: PSD(D<level>) - PSD(D1)."""
    d1 = np.load("Data/D1_train.npz")["d"]
    dn = loadmat(f"Data/D{level}.mat")["d"].ravel()
    f1, p1 = _psd(d1)
    f, pn = _psd(dn)
    p1 = np.interp(f, f1, p1)                  # line up with D<level>'s bands
    return f, np.clip(pn - p1, 0, None)        # power can't be negative


def generate_noise(level, n_samples, seed=None):
    """New noise shaped like D<level>'s extra noise. No seed = different every call."""
    rng = np.random.default_rng(time.time_ns() if seed is None else seed)
    white = rng.standard_normal(n_samples)     # equal power at every frequency
    f, extra = _extra_noise_psd(level)
    freqs = np.fft.rfftfreq(n_samples, d=1 / FS)
    target = np.interp(freqs, f, extra)        # target PSD at each FFT frequency
    target[0] = 0                              # no constant offset
    spectrum = np.fft.rfft(white) * np.sqrt(target * FS / 2)  # white noise PSD is 2/FS
    return np.fft.irfft(spectrum, n=n_samples)


# ---- scoring --------------------------------------------------------------
def pick_peaks(outputs, width=WINDOW, threshold=0.0):
    """Greedy peak picking: find the group of `width` scores with the biggest
    sum, guess its middle, zero the group, repeat until no group's average is
    above threshold. Returns guesses (sorted) and each guess's group average."""
    point = (width + 1) // 2
    scores = np.array(outputs, dtype=np.float64)
    kernel = np.ones(width)
    sums = np.convolve(scores, kernel, mode="valid")  # sums[i] = scores[i:i+width].sum()
    limit = threshold * width

    guesses, peak_scores = [], []
    while True:
        i = int(np.argmax(sums))
        if sums[i] <= limit:
            break
        guesses.append(i + point - 1)
        peak_scores.append(sums[i] / width)
        scores[i:i + width] = 0
        lo, hi = max(0, i - width + 1), min(len(sums), i + width)  # sums that changed
        sums[lo:hi] = np.convolve(scores[lo:hi + width - 1], kernel, mode="valid")

    guesses = np.array(guesses, dtype=np.int64)
    order = np.argsort(guesses)
    return guesses[order], np.array(peak_scores)[order]


def count_hits(real, guesses, tolerance=TOLERANCE):
    """One-to-one matching, closest pairs first, within ±tolerance. Returns hits."""
    order = np.argsort(guesses)
    sorted_guesses = guesses[order]
    pairs = []                                 # (distance, real position, guess position)
    for r, spike in enumerate(real):
        lo = np.searchsorted(sorted_guesses, spike - tolerance, side="left")
        hi = np.searchsorted(sorted_guesses, spike + tolerance, side="right")
        for k in range(lo, hi):
            pairs.append((abs(int(sorted_guesses[k]) - int(spike)), r, int(order[k])))
    pairs.sort()
    real_used = np.zeros(len(real), dtype=bool)
    guess_used = np.zeros(len(guesses), dtype=bool)
    hits = 0
    for _, r, g in pairs:
        if not real_used[r] and not guess_used[g]:
            real_used[r] = guess_used[g] = True
            hits += 1
    return hits


def best_f(outputs, real):
    """Peak-pick once, then try every threshold. Returns the best F and its
    threshold, precision and recall."""
    guesses, scores = pick_peaks(outputs)
    best = {"threshold": 0.0, "f": 0.0, "precision": 0.0, "recall": 0.0}
    for t in np.arange(0, 1 + SWEEP_STEP / 2, SWEEP_STEP):
        kept = guesses[scores > t]
        hits = count_hits(real, kept)
        if hits == 0:
            continue
        p, r = hits / len(kept), hits / len(real)
        f = 2 * p * r / (p + r)
        if f > best["f"]:
            best = {"threshold": round(float(t), 6), "f": f, "precision": p, "recall": r}
    return best


# ---- data and network -----------------------------------------------------
def make_labels(index, n_windows, tolerance=(WINDOW - 1) // 2):
    """1 for every window that starts within ±tolerance of a spike, else 0."""
    y = np.zeros(n_windows, dtype=np.float32)
    for offset in range(-tolerance, tolerance + 1):
        starts = index + offset
        starts = starts[(starts >= 0) & (starts < n_windows)]
        y[starts] = 1
    return y


def windows(signal, width=SHAPE[0]):
    """Every window of length `width`, one per sample. Shape (n_windows, width)."""
    return torch.tensor(signal, dtype=torch.float32).unfold(0, width, 1)


def make_model(shape=SHAPE):
    """MLP: Linear layers with ReLU between, no ReLU after the output."""
    layers = []
    for a, b in zip(shape[:-1], shape[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


def predict(model, x):
    """Sigmoid output (0 to 1) for every window."""
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(1)).numpy()


def val_signal(level, d):
    """The fixed noisy validation signal for a level (same every time)."""
    return d[SPLIT:] + generate_noise(level, len(d), seed=VAL_SEED + level)[SPLIT:]


# ---- training -------------------------------------------------------------
def train_run(target, level, hours, d, index, cls):
    name = f"class{target}_level{level}"
    out = f"{OUT_DIR}/{name}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    is_target = cls == target
    train_d = d[:SPLIT]
    train_index = index[is_target & (index < SPLIT)]
    val_index = index[is_target & (index >= SPLIT)] - SPLIT
    y_train = torch.tensor(make_labels(train_index, len(train_d) - SHAPE[0] + 1))
    x_val = windows(val_signal(level, d))

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(POS_WEIGHT)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name}: {'-'.join(map(str, SHAPE))} ({n_params:,} weights), "
          f"{len(train_index)} training / {len(val_index)} validation spikes, "
          f"{hours} h ===", flush=True)

    n_train = len(y_train)
    best, warned = -1.0, False
    start, deadline, epoch = time.time(), time.time() + hours * 3600, 0
    while time.time() < deadline:
        epoch += 1
        x_train = windows(train_d + generate_noise(level, len(train_d)))  # new noise
        order = torch.tensor(rng.permutation(n_train))

        total = 0.0
        for i in range(0, n_train, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(x_train[b]).squeeze(1), y_train[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        loss = total / n_train

        if epoch % CHECK_EVERY == 0:
            result = best_f(predict(model, x_val), val_index)
            row = {"epoch": epoch, "minutes": round((time.time() - start) / 60, 2),
                   "loss": loss, **result, "n_params": n_params}
            with open(f"{out}/log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if result["f"] > best:
                best = result["f"]
                torch.save(model.state_dict(), f"{out}/best.pt")
            print(f"{name}  epoch {epoch:5d}  {row['minutes']:6.1f} min  loss {loss:.4f}  "
                  f"F {result['f']:.3f} at threshold {result['threshold']:.2f}", flush=True)
            if epoch >= COLLAPSE_EPOCH and best < COLLAPSE_F and not warned:
                warned = True
                print(f"WARNING {name}: best F is still {best:.3f} after {epoch} epochs. "
                      f"The network may have collapsed to 'never a spike'.", flush=True)

    torch.save(model.state_dict(), f"{out}/last.pt")
    print(f"Done {name}: {epoch} epochs, best F {best:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="One detector per class per level.")
    parser.add_argument("hours", type=float, help="training time per run")
    parser.add_argument("classes", type=int, nargs="*", default=list(CLASSES))
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    args = parser.parse_args()

    torch.set_num_threads(THREADS)
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load("Data/D1_train.npz")
    d, index, cls = data["d"], data["Index"], data["Class"]
    print(f"Classes {args.classes}, levels {args.levels}, {args.hours} h each "
          f"= {len(args.classes) * len(args.levels) * args.hours:g} h", flush=True)

    for target in args.classes:
        for level in args.levels:
            train_run(target, level, args.hours, d, index, cls)


if __name__ == "__main__":
    main()
