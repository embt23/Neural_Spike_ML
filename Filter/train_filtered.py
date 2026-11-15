"""Does band-pass filtering help the detector at level 6?

Trains the same small detector as Pos_weight/Test_POS_WEIGHT.py
(50-50-25-1, POS_WEIGHT 5, 100 epochs, fresh noise every epoch), but every
noisy signal is band-pass filtered (300 to 3000 Hz) before it is cut into
windows. Then tests it like Pos_weight/fresh_noise_test.py: 20 new noisy
validation signals, threshold swept 0 to 1, F averaged, peak reported.

Run from the project root:
    python Filter/train_filtered.py
Saves to Filter/data/: the model (.pt) and the averaged curves (.npz).
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "Pos_weight")
sys.path.insert(0, "Modular")
sys.path.insert(0, "Other")
from Test_POS_WEIGHT import make_labels, make_model, windows, predict, SPLIT, WIDTH  # noqa: E402
from generate_noise import generate_noise                                          # noqa: E402
from filtering import bandpass                                                     # noqa: E402
from find_threshold import sweep                                                   # noqa: E402

LEVEL = 6
POS_WEIGHT = 5
EPOCHS = 100
BATCH = 256
LR = 0.001
N_GRAPHS = 20
SEED = 0
OUT_DIR = "Filter/data"


def train(train_d, train_index):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    n_train = len(train_d) - WIDTH + 1
    y_train = torch.tensor(make_labels(train_index, n_train))

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(POS_WEIGHT)))
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # new noise every epoch, then filter the noisy signal (like real data)
        x_train = windows(bandpass(train_d + generate_noise(LEVEL, len(train_d))))
        order = torch.tensor(rng.permutation(n_train))
        total = 0.0
        for i in range(0, n_train, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(x_train[b]).squeeze(1), y_train[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}/{EPOCHS}  {(time.time() - start) / 60:4.1f} min  "
                  f"loss {total / n_train:.4f}", flush=True)
    return model


def test(model, clean_val, val_index):
    """Average the F curve over N_GRAPHS new noisy, filtered validation signals."""
    curves = {"f": [], "precision": [], "recall": []}
    for g in range(N_GRAPHS):
        noisy = clean_val + generate_noise(LEVEL, SPLIT + len(clean_val))[SPLIT:]
        res = sweep(predict(model, windows(bandpass(noisy))), val_index)
        for k in curves:
            curves[k].append(res[k])
    curves = {k: np.array(v) for k, v in curves.items()}
    return res["thresholds"], curves


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load("Data/D1_train.npz")
    d, index = data["d"], data["Index"]
    train_d, clean_val = d[:SPLIT], d[SPLIT:]
    train_index = index[index < SPLIT]
    val_index = index[index >= SPLIT] - SPLIT

    print(f"Level {LEVEL}, POS_WEIGHT {POS_WEIGHT}, band-pass 300-3000 Hz", flush=True)
    model = train(train_d, train_index)
    torch.save(model.state_dict(), f"{OUT_DIR}/level{LEVEL}_pw{POS_WEIGHT}_filtered.pt")

    thresholds, c = test(model, clean_val, val_index)
    mean_f = c["f"].mean(axis=0)
    b = int(np.argmax(mean_f))
    np.savez_compressed(f"{OUT_DIR}/level{LEVEL}_pw{POS_WEIGHT}_filtered.npz",
                        thresholds=thresholds, mean_f=mean_f, std_f=c["f"].std(axis=0),
                        mean_precision=c["precision"].mean(axis=0),
                        mean_recall=c["recall"].mean(axis=0))
    print(f"\nFiltered, averaged over {N_GRAPHS} graphs: F {mean_f[b]:.3f} ±{c['f'][:, b].std():.3f} "
          f"at threshold {thresholds[b]:.3f}  (precision {c['precision'].mean(axis=0)[b]:.3f}, "
          f"recall {c['recall'].mean(axis=0)[b]:.3f})")

    old = np.load("Pos_weight/data/fresh_noise/summary.npz")[f"level{LEVEL}_pw{POS_WEIGHT}_peak"]
    print(f"Unfiltered (run 1-20), same test:      F {old[1]:.3f} at threshold {old[0]:.3f}")


if __name__ == "__main__":
    main()
