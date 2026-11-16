"""Spike detector shapes: which hidden-layer shape detects level 6 spikes best?

Same set-up as the overnight run (Big_model/train_big.py): windows of 50
samples, STEP 1, labels within ±4 of a spike, POS_WEIGHT 5, raw noisy input,
fresh level 6 noise every epoch. Only the hidden layers change.

Validation uses the overnight run's saved signal (Big_model/data/val_signal.npz),
so these F-scores compare directly with Big_model/data/log.jsonl.

Run from the project root: hours per shape, then one or more hidden shapes:
    python Detect_shapes/train_detector.py 2 100-100-25 300 70-70-70-70

Saves to Detect_shapes/data/<shape>/:
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

sys.path.insert(0, "Pos_weight")
sys.path.insert(0, "Modular")
sys.path.insert(0, "Other")
from Test_POS_WEIGHT import make_labels, windows, predict, SPLIT  # noqa: E402
from generate_noise import generate_noise                          # noqa: E402
from find_threshold import sweep                                   # noqa: E402

# ---- settings -------------------------------------------------------------
LEVEL = 6
POS_WEIGHT = 5
WIDTH = 50                 # must match Test_POS_WEIGHT.py (windows() uses its WIDTH)
BATCH = 256
LR = 0.001
CHECK_EVERY = 25           # epochs between validation checks
SWEEP_STEP = 0.01
THREADS = 4                # CPU cores (the denoiser run uses the other 4)
SEED = 0
OUT_DIR = "Detect_shapes/data"
VAL_FILE = "Big_model/data/val_signal.npz"


def make_model(hidden):
    """MLP: WIDTH inputs -> hidden layers (ReLU) -> 1 output (spike logit)."""
    layers, size = [], WIDTH
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def check(model, x_val, val_index):
    """Best F on the validation signal (threshold swept from 0 to 1)."""
    res = sweep(predict(model, x_val), val_index, step=SWEEP_STEP)
    best = int(np.argmax(res["f"]))
    return {"threshold": float(res["thresholds"][best]), "f": float(res["f"][best]),
            "precision": float(res["precision"][best]), "recall": float(res["recall"][best])}


def train_shape(hidden, hours, train_d, y_train, x_val, val_index):
    name = "-".join(map(str, hidden))
    out = f"{OUT_DIR}/{name}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    model = make_model(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(POS_WEIGHT)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Detector {WIDTH}-{name}-1 ({n_params:,} weights), {hours} h ===", flush=True)

    n_train = len(y_train)
    best = -1.0
    start, deadline, epoch = time.time(), time.time() + hours * 3600, 0
    while time.time() < deadline:
        epoch += 1
        x_train = windows(train_d + generate_noise(LEVEL, len(train_d)))  # new noise
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
            print(f"epoch {epoch:5d}  {row['minutes']:6.1f} min  loss {loss:.4f}  "
                  f"F {result['f']:.3f} at threshold {result['threshold']:.2f}", flush=True)

    torch.save(model.state_dict(), f"{out}/last.pt")
    print(f"Done {name}: {epoch} epochs, best F {best:.3f}", flush=True)


def main():
    hours = float(sys.argv[1])
    shapes = [[int(h) for h in s.split("-")] for s in sys.argv[2:]]
    torch.set_num_threads(THREADS)
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load("Data/D1_train.npz")
    train_d, index = data["d"][:SPLIT], data["Index"]
    y_train = torch.tensor(make_labels(index[index < SPLIT], len(train_d) - WIDTH + 1))
    val = np.load(VAL_FILE)
    x_val = windows(val["val_d"].astype(np.float64))
    for hidden in shapes:
        train_shape(hidden, hours, train_d, y_train, x_val, val["val_index"])


if __name__ == "__main__":
    main()
