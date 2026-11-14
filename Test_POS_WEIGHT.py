"""Spike detector experiment: how do POS_WEIGHT and noise level affect detection?

For each noise level, one detector is trained per POS_WEIGHT. Every 10th epoch
each detector is run over every window of the validation signal, and the raw
output vector (0 to 1 per window) is saved. Turning those vectors into spike
guesses and F-scores is done later, in a separate script.

Run from the project root, choosing the levels (default 1 2 3):
    python Test_POS_WEIGHT.py 1 2 3
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "Other")
from generate_noise import generate_noise  # noqa: E402

# ---- settings -------------------------------------------------------------
WIDTH = 50                            # samples per window (2 ms)
STEP = 1                              # a window starts at every sample
TOLERANCE = 4                         # window is a spike if it starts within ±4 of an Index
HIDDEN = [50, 25]                     # hidden layer sizes
POS_WEIGHTS = [10, 25, 65, 100, 300]
EPOCHS = 100
BATCH = 256
LR = 0.001
SAVE_EVERY = 10                       # save validation outputs every 10th epoch
SPLIT = 1_000_000                     # train = D1_train[:SPLIT], validation = the rest
SEED = 0
OUT_DIR = "Results/pos_weight"


def make_labels(index, n_windows):
    """1 for every window that starts within ±TOLERANCE of a spike, else 0."""
    y = np.zeros(n_windows, dtype=np.float32)
    for offset in range(-TOLERANCE, TOLERANCE + 1):
        starts = index + offset
        starts = starts[(starts >= 0) & (starts < n_windows)]
        y[starts] = 1
    return y


def make_model():
    """MLP: WIDTH inputs -> HIDDEN layers (ReLU) -> 1 output (spike logit)."""
    layers, size = [], WIDTH
    for h in HIDDEN:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def windows(signal):
    """Every window of length WIDTH, one per STEP. Shape (n_windows, WIDTH)."""
    return torch.tensor(signal, dtype=torch.float32).unfold(0, WIDTH, STEP)


def predict(model, x):
    """Sigmoid output for every window, as float16 to save space."""
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(1)).numpy().astype(np.float16)


def run_level(level, d, index, cls):
    print(f"\n=== Level {level} ===")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # split D1_train into training and validation parts
    train_d, val_d = d[:SPLIT], d[SPLIT:]
    is_val = index >= SPLIT
    train_index = index[~is_val]
    val_index, val_class = index[is_val] - SPLIT, cls[is_val]

    # validation noise: generated once, the same for every epoch and POS_WEIGHT
    if level > 1:
        val_d = val_d + generate_noise(level, len(d))[SPLIT:]
    x_val = windows(val_d)

    n_train = len(train_d) - WIDTH + 1
    y_train = torch.tensor(make_labels(train_index, n_train))

    # one model per POS_WEIGHT, all starting from the same weights
    models, opts, losses = [], [], []
    for pw in POS_WEIGHTS:
        torch.manual_seed(SEED)
        m = make_model()
        models.append(m)
        opts.append(torch.optim.Adam(m.parameters(), lr=LR))
        losses.append(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pw))))

    saved_epochs = []
    outputs = {pw: [] for pw in POS_WEIGHTS}
    train_loss = {pw: [] for pw in POS_WEIGHTS}
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # training noise: new each epoch, shared by every POS_WEIGHT
        noisy = train_d + generate_noise(level, len(train_d)) if level > 1 else train_d
        x_train = windows(noisy)
        order = torch.tensor(rng.permutation(n_train))  # same shuffle for every model

        totals = [0.0] * len(models)
        for i in range(0, n_train, BATCH):
            b = order[i:i + BATCH]
            xb, yb = x_train[b], y_train[b]
            for k, (m, opt, loss_fn) in enumerate(zip(models, opts, losses)):
                opt.zero_grad()
                loss = loss_fn(m(xb).squeeze(1), yb)
                loss.backward()
                opt.step()
                totals[k] += loss.item() * len(b)

        for pw, total in zip(POS_WEIGHTS, totals):
            train_loss[pw].append(total / n_train)

        elapsed = time.time() - start
        eta = elapsed / epoch * (EPOCHS - epoch)
        print(f"epoch {epoch:3d}/{EPOCHS}  {elapsed / 60:5.1f} min  "
              f"(level ETA {eta / 60:.1f} min)  loss "
              + "  ".join(f"pw{pw}={train_loss[pw][-1]:.4f}" for pw in POS_WEIGHTS))

        if epoch % SAVE_EVERY == 0:
            saved_epochs.append(epoch)
            for pw, m in zip(POS_WEIGHTS, models):
                outputs[pw].append(predict(m, x_val))
                np.savez_compressed(
                    f"{OUT_DIR}/level{level}_pw{pw}.npz",
                    outputs=np.stack(outputs[pw]),   # (n_saved_epochs, n_val_windows)
                    epochs=np.array(saved_epochs),
                    train_loss=np.array(train_loss[pw]),
                    val_index=val_index,             # 0-based, relative to val_d
                    val_class=val_class,
                    val_d=val_d.astype(np.float32),  # the noisy validation signal
                    level=level, pos_weight=pw, width=WIDTH, step=STEP,
                    tolerance=TOLERANCE, hidden=np.array(HIDDEN), lr=LR,
                    batch=BATCH, seed=SEED,
                )
                torch.save(m.state_dict(), f"{OUT_DIR}/level{level}_pw{pw}.pt")
            print(f"  saved validation outputs for epoch {epoch}")


def main():
    levels = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load("Data/D1_train.npz")
    for level in levels:
        run_level(level, data["d"], data["Index"], data["Class"])


if __name__ == "__main__":
    main()
