"""Single-class detector: one network per noise level that finds ONE class.

A window is labelled 1 if it starts within ±4 samples of a spike of the
chosen class (acceptance window 9); every other window, including ones on
spikes of the other classes, is 0. Network 50-200-100-50-1, POS_WEIGHT 25
(5 x the all-spike detectors' 5, because one class has 5 times fewer spikes).

Training noise: brand new every epoch, so the network learns to ignore noise
in general, not one particular noise pattern.
Validation noise: one fixed signal per level (seed VAL_SEED + level), so the
checks always score on the same signal and runs for different classes can be
compared. Rebuild it any time with val_signal(level).

Scoring: peak picking (group of 9), then F against the chosen class's spikes
only (±50 samples), threshold swept from 0 to 1.

Run from the project root: hours per level, the class, then the levels
(default 2 3 4 5 6):
    python Class3/train_class3.py 1 3
    python Class3/train_class3.py 1 4

Saves to Class3/data/class<C>_level<L>/:
  log.jsonl   one line per check: epoch, minutes, loss, best threshold, F, P, R
  best.pt     the model with the best validation F so far
  last.pt     the model at the end
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "Modular")
sys.path.insert(0, "Other")
from generate_noise import generate_noise   # noqa: E402
from find_threshold import sweep            # noqa: E402

# ---- settings -------------------------------------------------------------
SHAPE = [50, 200, 100, 50, 1]   # first number = input window length (2 ms)
WINDOW = 9                 # acceptance window: ±4 samples, also the peak-picking group
POS_WEIGHT = 25
BATCH = 256
LR = 0.001
CHECK_EVERY = 25           # epochs between validation checks
SWEEP_STEP = 0.01
COLLAPSE_EPOCH = 100       # warn if F is still below COLLAPSE_F at this epoch
COLLAPSE_F = 0.1
THREADS = 4                # CPU cores per run (run two terminals side by side)
SPLIT = 1_000_000          # train = D1_train[:SPLIT], validation = the rest
SEED = 0
VAL_SEED = 1000            # validation noise seed = VAL_SEED + level
OUT_DIR = "Class3/data"


def make_labels(index, n_windows, tolerance):
    """1 for every window that starts within ±tolerance of a spike, else 0."""
    y = np.zeros(n_windows, dtype=np.float32)
    for offset in range(-tolerance, tolerance + 1):
        starts = index + offset
        starts = starts[(starts >= 0) & (starts < n_windows)]
        y[starts] = 1
    return y


def make_model(shape=SHAPE):
    """MLP from a shape like [50, 100, 25, 1]: Linear layers, ReLU between."""
    layers = []
    for a, b in zip(shape[:-1], shape[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])     # no ReLU after the output layer


def windows(signal, width=SHAPE[0]):
    """Every window of length `width`, one per sample. Shape (n_windows, width)."""
    return torch.tensor(signal, dtype=torch.float32).unfold(0, width, 1)


def predict(model, x):
    """Sigmoid output for every window."""
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(1)).numpy()


def val_signal(level):
    """The fixed noisy validation signal for a level (same every time)."""
    d = np.load("Data/D1_train.npz")["d"]
    return d[SPLIT:] + generate_noise(level, len(d), seed=VAL_SEED + level)[SPLIT:]


def check(model, x_val, val_index):
    """Best F against the chosen class's spikes (threshold swept from 0 to 1)."""
    res = sweep(predict(model, x_val), val_index, step=SWEEP_STEP, width=WINDOW)
    best = int(np.argmax(res["f"]))
    return {"threshold": float(res["thresholds"][best]), "f": float(res["f"][best]),
            "precision": float(res["precision"][best]), "recall": float(res["recall"][best])}


def train_level(target, level, hours, train_d, y_train, val_index):
    name = f"class{target}_level{level}"
    out = f"{OUT_DIR}/{name}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    x_val = windows(val_signal(level))

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(POS_WEIGHT)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name}: {'-'.join(map(str, SHAPE))} ({n_params:,} weights), "
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
            result = check(model, x_val, val_index)
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
    hours = float(sys.argv[1])
    target = int(sys.argv[2])
    levels = [int(a) for a in sys.argv[3:]] or [2, 3, 4, 5, 6]
    torch.set_num_threads(THREADS)
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load("Data/D1_train.npz")
    d, index, cls = data["d"], data["Index"], data["Class"]
    is_target = cls == target
    train_d = d[:SPLIT]
    train_index = index[is_target & (index < SPLIT)]
    val_index = index[is_target & (index >= SPLIT)] - SPLIT
    y_train = torch.tensor(make_labels(train_index, len(train_d) - SHAPE[0] + 1,
                                       (WINDOW - 1) // 2))
    print(f"Class {target}: {len(train_index)} training spikes, "
          f"{len(val_index)} validation spikes, levels {levels}", flush=True)

    for level in levels:
        train_level(target, level, hours, train_d, y_train, val_index)


if __name__ == "__main__":
    main()
