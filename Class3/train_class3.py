"""Class 3 detector: which acceptance window and network shape find class 3 best?

Level 6 noise only. A window is labelled 1 if it starts close to a CLASS 3
spike; every other window (including ones on spikes of classes 1, 2, 4, 5)
is 0. Fresh training noise every epoch, POS_WEIGHT 5, raw noisy input.

Acceptance window W (5, 7, 9 or 13): a window counts as a spike if it starts
within ±(W-1)/2 samples of a class 3 spike (W=5 -> 2 before and 2 after).
Peak picking uses the same W as its group size.

Shapes are written input-hidden...-output. The first number is the input
window length (50 = 2 ms, 25 = 1 ms).

Validation: the same level 6 signal as the earlier runs
(Big_model/data/val_signal.npz), scored against class 3 spikes only
(±50 samples, threshold swept from 0 to 1).

Run from the project root: hours per run, then one or more acceptance windows.
Every shape is trained for each window:
    python Class3/train_class3.py 1 5 7

Saves to Class3/data/w<W>_<shape>/:
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
LEVEL = 6
TARGET_CLASS = 3
POS_WEIGHT = 5
SHAPES = [
    [50, 200, 100, 70, 50, 25, 5, 1],
    [50, 100, 100, 50, 25, 5, 1],
    [25, 50, 100, 200, 100, 50, 25, 5, 1],
]
BATCH = 256
LR = 0.001
CHECK_EVERY = 25           # epochs between validation checks
SWEEP_STEP = 0.01
THREADS = 4                # CPU cores per run (run two terminals side by side)
SPLIT = 1_000_000          # train = D1_train[:SPLIT], validation = the rest
SEED = 0
OUT_DIR = "Class3/data"
VAL_FILE = "Big_model/data/val_signal.npz"


def make_labels(index, n_windows, tolerance):
    """1 for every window that starts within ±tolerance of a spike, else 0."""
    y = np.zeros(n_windows, dtype=np.float32)
    for offset in range(-tolerance, tolerance + 1):
        starts = index + offset
        starts = starts[(starts >= 0) & (starts < n_windows)]
        y[starts] = 1
    return y


def make_model(shape):
    """MLP from a shape like [50, 100, 25, 1]: Linear layers, ReLU between."""
    layers = []
    for a, b in zip(shape[:-1], shape[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])     # no ReLU after the output layer


def windows(signal, width):
    """Every window of length `width`, one per sample. Shape (n_windows, width)."""
    return torch.tensor(signal, dtype=torch.float32).unfold(0, width, 1)


def predict(model, x):
    """Sigmoid output for every window."""
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(1)).numpy()


def check(model, x_val, val_index, window):
    """Best F against the class 3 spikes (threshold swept from 0 to 1)."""
    res = sweep(predict(model, x_val), val_index, step=SWEEP_STEP, width=window)
    best = int(np.argmax(res["f"]))
    return {"threshold": float(res["thresholds"][best]), "f": float(res["f"][best]),
            "precision": float(res["precision"][best]), "recall": float(res["recall"][best])}


def train_run(shape, window, hours, train_d, train_index, val_d, val_index):
    name = f"w{window}_{'-'.join(map(str, shape))}"
    out = f"{OUT_DIR}/{name}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    width = shape[0]
    n_train = len(train_d) - width + 1
    y_train = torch.tensor(make_labels(train_index, n_train, (window - 1) // 2))
    x_val = windows(val_d, width)

    model = make_model(shape)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(POS_WEIGHT)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} ({n_params:,} weights), {hours} h ===", flush=True)

    best = -1.0
    start, deadline, epoch = time.time(), time.time() + hours * 3600, 0
    while time.time() < deadline:
        epoch += 1
        x_train = windows(train_d + generate_noise(LEVEL, len(train_d)), width)  # new noise
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
            result = check(model, x_val, val_index, window)
            row = {"epoch": epoch, "minutes": round((time.time() - start) / 60, 2),
                   "loss": loss, **result, "n_params": n_params}
            with open(f"{out}/log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if result["f"] > best:
                best = result["f"]
                torch.save(model.state_dict(), f"{out}/best.pt")
            print(f"{name}  epoch {epoch:5d}  {row['minutes']:6.1f} min  loss {loss:.4f}  "
                  f"F {result['f']:.3f} at threshold {result['threshold']:.2f}", flush=True)

    torch.save(model.state_dict(), f"{out}/last.pt")
    print(f"Done {name}: {epoch} epochs, best F {best:.3f}", flush=True)


def main():
    hours = float(sys.argv[1])
    window_list = [int(a) for a in sys.argv[2:]]
    for w in window_list:
        if w % 2 == 0:
            raise ValueError(f"acceptance window must be odd, got {w}")
    torch.set_num_threads(THREADS)
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load("Data/D1_train.npz")
    d, index, cls = data["d"], data["Index"], data["Class"]
    target = cls == TARGET_CLASS
    train_d = d[:SPLIT]
    train_index = index[target & (index < SPLIT)]
    val_index = index[target & (index >= SPLIT)] - SPLIT
    val_d = np.load(VAL_FILE)["val_d"].astype(np.float64)
    print(f"Class {TARGET_CLASS}: {len(train_index)} training spikes, "
          f"{len(val_index)} validation spikes", flush=True)

    for window in window_list:
        for shape in SHAPES:
            train_run(shape, window, hours, train_d, train_index, val_d, val_index)


if __name__ == "__main__":
    main()
