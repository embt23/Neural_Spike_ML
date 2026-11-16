"""Denoiser: from the level 6 noisy signal, predict the original D1 signal.

For every sample, the network sees 61 points of the noisy signal around it
(dense from -20 to +20, sparser further out to ±100) and outputs one number:
its guess of the original (D1) value at that sample. The output node is
linear, so the guess is in the signal's own units (D1 runs from about -2.3
to 9.7). Loss: mean squared error. Score: RMS = sqrt(mean squared error).

Training: D1_train[:SPLIT] plus brand-new level 6 noise every epoch.
Validation: D1_train[SPLIT:] plus the noise saved by the overnight run
(Big_model/data/val_signal.npz), so every model is scored on the same signal.

Run from the project root: hours per shape, then one or more hidden shapes:
    python Denoise/train_denoise.py 2 100-100-25 300 70-70-70-70

Saves to Denoise/data/<shape>/:
  log.jsonl   one line per check: epoch, minutes, train RMS, validation RMS
              (all samples, near spikes, away from spikes)
  best.pt     the model with the lowest validation RMS so far
  last.pt     the model at the end
and Denoise/data/baselines.json: RMS of simple non-network guesses.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "Modular")
from generate_noise import generate_noise  # noqa: E402
from filtering import bandpass             # noqa: E402

# ---- settings -------------------------------------------------------------
LEVEL = 6
SPLIT = 1_000_000          # same split as Pos_weight/Test_POS_WEIGHT.py
FAR = [100, 80, 60, 55, 50, 45, 40, 35, 30, 25]   # sparse taps, each side
OFFSETS = np.array(sorted([-f for f in FAR] + list(range(-20, 21)) + FAR))  # 61 taps
REACH = 100                # furthest tap, so the first/last 100 samples are skipped
SPIKE_LEN = 50             # "near a spike" = within Index to Index+50
BATCH = 256
LR = 0.001
CHECK_EVERY = 5            # epochs between validation checks
THREADS = 4                # CPU cores (the detector run uses the other 4)
SEED = 0
OUT_DIR = "Denoise/data"
VAL_FILE = "Big_model/data/val_signal.npz"


def make_model(hidden):
    """MLP: 61 inputs -> hidden layers (ReLU) -> 1 linear output."""
    layers, size = [], len(OFFSETS)
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def gather(signal, centres):
    """The 61 taps around each centre. signal: 1D tensor. Shape (len(centres), 61)."""
    return signal[centres[:, None] + torch.from_numpy(OFFSETS)]


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def predict(model, signal, centres):
    with torch.no_grad():
        return model(gather(signal, centres)).squeeze(1).numpy()


def load_data():
    data = np.load("Data/D1_train.npz")
    d, index = data["d"], data["Index"]
    val = np.load(VAL_FILE)
    clean_val = d[SPLIT:]
    noisy_val = val["val_d"].astype(np.float64)
    val_index = val["val_index"]

    # validation samples that have all 61 taps, and which of them are near a spike
    val_centres = np.arange(REACH, len(clean_val) - REACH)
    near = np.zeros(len(clean_val), dtype=bool)
    for i in val_index:
        near[i:i + SPIKE_LEN] = True
    near = near[val_centres]
    return d[:SPLIT], clean_val, noisy_val, val_centres, near


def baselines(clean_val, noisy_val, val_centres, near):
    """RMS of guesses that use no network, to compare the networks against."""
    c = clean_val[val_centres]
    guesses = {
        "noisy input (do nothing)": noisy_val[val_centres],
        "constant (D1 mean)": np.full(len(c), clean_val.mean()),
        "band-pass 300-3000 Hz": bandpass(noisy_val)[val_centres],
    }
    out = {name: {"all": rms(g, c), "near_spike": rms(g[near], c[near]),
                  "away": rms(g[~near], c[~near])} for name, g in guesses.items()}
    with open(f"{OUT_DIR}/baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    for name, r in out.items():
        print(f"baseline {name:26s} RMS {r['all']:.3f}  near spike {r['near_spike']:.3f}  "
              f"away {r['away']:.3f}", flush=True)


def train_shape(hidden, hours, train_d, clean_val, noisy_val, val_centres, near):
    name = "-".join(map(str, hidden))
    out = f"{OUT_DIR}/{name}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    model = make_model(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Denoiser {len(OFFSETS)}-{name}-1 ({n_params:,} weights), {hours} h ===",
          flush=True)

    clean_t = torch.tensor(train_d, dtype=torch.float32)
    noisy_val_t = torch.tensor(noisy_val, dtype=torch.float32)
    clean_c = clean_val[val_centres]
    centres = torch.arange(REACH, len(train_d) - REACH)
    best = np.inf

    start, deadline, epoch = time.time(), time.time() + hours * 3600, 0
    while time.time() < deadline:
        epoch += 1
        noisy_t = torch.tensor(train_d + generate_noise(LEVEL, len(train_d)),
                               dtype=torch.float32)          # new noise every epoch
        order = centres[torch.from_numpy(rng.permutation(len(centres)))]

        total = 0.0
        for i in range(0, len(order), BATCH):
            c = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(gather(noisy_t, c)).squeeze(1), clean_t[c])
            loss.backward()
            opt.step()
            total += loss.item() * len(c)
        train_rms = float(np.sqrt(total / len(order)))

        if epoch % CHECK_EVERY == 0:
            p = predict(model, noisy_val_t, torch.from_numpy(val_centres))
            row = {"epoch": epoch, "minutes": round((time.time() - start) / 60, 2),
                   "train_rms": train_rms, "val_rms": rms(p, clean_c),
                   "val_rms_near_spike": rms(p[near], clean_c[near]),
                   "val_rms_away": rms(p[~near], clean_c[~near]),
                   "n_params": n_params}
            with open(f"{out}/log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if row["val_rms"] < best:
                best = row["val_rms"]
                torch.save(model.state_dict(), f"{out}/best.pt")
            print(f"epoch {epoch:5d}  {row['minutes']:6.1f} min  train {train_rms:.3f}  "
                  f"val {row['val_rms']:.3f}  (near spike {row['val_rms_near_spike']:.3f}, "
                  f"away {row['val_rms_away']:.3f})", flush=True)

    torch.save(model.state_dict(), f"{out}/last.pt")
    print(f"Done {name}: {epoch} epochs, best val RMS {best:.3f}", flush=True)


def main():
    hours = float(sys.argv[1])
    shapes = [[int(h) for h in s.split("-")] for s in sys.argv[2:]]
    torch.set_num_threads(THREADS)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_d, clean_val, noisy_val, val_centres, near = load_data()
    baselines(clean_val, noisy_val, val_centres, near)
    for hidden in shapes:
        train_shape(hidden, hours, train_d, clean_val, noisy_val, val_centres, near)


if __name__ == "__main__":
    main()
