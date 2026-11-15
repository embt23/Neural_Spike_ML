"""Overnight run: one bigger spike detector, 50-100-50-25-1, on level 6 noise.

Same set-up as Pos_weight/Test_POS_WEIGHT.py (windows of 50 samples, STEP 1,
labels within ±4 of a spike, fresh training noise every epoch), but:
  - one model only: hidden layers 100, 50, 25 and POS_WEIGHT 5
  - it trains until HOURS have passed instead of a fixed number of epochs
  - every CHECK_EVERY epochs it saves the model and checks it on the
    validation signal (sweeps the threshold and records the best F)

Run from the project root (the hours are optional, default 9):
    python Big_model/train_big.py 9

Saves to Big_model/data/:
  log.jsonl          one line per check: epoch, time, loss, best threshold, F
  epoch<NNNNN>.pt    the model at each check
  val_signal.npz     the noisy validation signal used for every check
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
HIDDEN = [100, 50, 25]
BATCH = 256
LR = 0.001
CHECK_EVERY = 25           # epochs between saves and validation checks
SWEEP_STEP = 0.01          # threshold step for the checks (coarse, to save time)
SEED = 0
OUT_DIR = "Big_model/data"


def make_model():
    """MLP: WIDTH inputs -> HIDDEN layers (ReLU) -> 1 output (spike logit)."""
    layers, size = [], WIDTH
    for h in HIDDEN:
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


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 9
    deadline = time.time() + hours * 3600
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    data = np.load("Data/D1_train.npz")
    d, index = data["d"], data["Index"]
    train_d, clean_val = d[:SPLIT], d[SPLIT:]
    train_index = index[index < SPLIT]
    val_index = index[index >= SPLIT] - SPLIT

    # validation noise: made once, so every check sees the same signal
    val_d = clean_val + generate_noise(LEVEL, len(d))[SPLIT:]
    np.savez_compressed(f"{OUT_DIR}/val_signal.npz", val_d=val_d.astype(np.float32),
                        val_index=val_index)
    x_val = windows(val_d)

    n_train = len(train_d) - WIDTH + 1
    y_train = torch.tensor(make_labels(train_index, n_train))

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(POS_WEIGHT)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Level {LEVEL}, POS_WEIGHT {POS_WEIGHT}, {WIDTH}-{'-'.join(map(str, HIDDEN))}-1 "
          f"({n_params:,} parameters), training for {hours} h", flush=True)

    start, epoch = time.time(), 0
    while time.time() < deadline:
        epoch += 1
        # training noise: new every epoch
        x_train = windows(train_d + generate_noise(LEVEL, len(train_d)))
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
            torch.save(model.state_dict(), f"{OUT_DIR}/epoch{epoch:05d}.pt")
            result = check(model, x_val, val_index)
            row = {"epoch": epoch, "minutes": round((time.time() - start) / 60, 1),
                   "loss": loss, **result}
            with open(f"{OUT_DIR}/log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"epoch {epoch:5d}  {row['minutes']:6.1f} min  loss {loss:.4f}  "
                  f"F {result['f']:.3f} at threshold {result['threshold']:.2f}", flush=True)

    print(f"Done: {epoch} epochs in {(time.time() - start) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
