"""Filter competition at level 6: does filtering the signal first help the networks?

Four competitors: unfiltered, band-pass, Wiener and wavelet. Each filter uses
its winning setting from tune_filters.py (data/tuning.json). Every competitor
goes through exactly the same steps, with the same networks, settings and
noise seeds. The only difference is the filter:

  1. Detector: 50-100-50-25-1 (Big_model/train_big.py), POS_WEIGHT 5,
     a fixed DET_EPOCHS epochs. Every epoch: clean training part + new noise,
     then the filter, then the network.
  2. Threshold: the peak-picking threshold with the best detection F,
     averaged over N_THRESH validation signals.
  3. Classifier data: the detector + threshold run on N_CLS training signals.
     Each detection is matched to the real spikes like the grader does
     (label 1-5 = class, 0 = fail). Its window (9 before to 70 after) is cut
     from the *filtered* signal, because that's what the classifier will see.
  4. Classifier: 80-100-80-40-6 (Classifier/train_classifier.py), a fixed
     CLS_EPOCHS epochs. No best-epoch picking, so no lucky checkpoints.
  5. Score: the whole pipeline on N_SCORE validation signals, graded with
     V0/grading.py (F_total = 0.3 F_detect + 0.7 F_class). Mean and spread.

Order everywhere: clean signal -> add noise -> filter -> network (real D6
already has its noise). Training uses D1_train[:SPLIT], validation the rest.
The test set is not touched. The threshold signals, the scoring signals and the
tuning signals (seeds 3000+) all use different seeds, so nothing is scored on
signals it was tuned on.

Run from the project root, choosing the competitors (default: all four):
    python Filter_compare/compete.py unfiltered bandpass wiener wavelet
Saves Filter_compare/data/<competitor>/ (detector.pt, classifier.pt,
result.json), then prints a table of every competitor that has finished.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "Pos_weight")
sys.path.insert(0, "Big_model")
sys.path.insert(0, "Classifier")
sys.path.insert(0, "Modular")
sys.path.insert(0, "Other")
sys.path.insert(0, "V0")
from Test_POS_WEIGHT import make_labels, windows, predict, SPLIT, WIDTH   # noqa: E402
from train_big import make_model as make_detector                         # noqa: E402
from train_classifier import make_model as make_classifier, classify      # noqa: E402
from make_dataset import detect_and_label                                 # noqa: E402
from generate_noise import generate_noise                                 # noqa: E402
from filtering import bandpass, wiener, wavelet_denoise                   # noqa: E402
from find_threshold import sweep                                          # noqa: E402
from grading import grade                                                 # noqa: E402

# ---- settings (the same for every competitor) -----------------------------
LEVEL = 6
POS_WEIGHT = 5
DET_EPOCHS = 100
CLS_EPOCHS = 300
BATCH = 256
LR = 0.001
SEED = 0                    # network starting weights and shuffle order
N_THRESH = 5                # validation signals for choosing the threshold
N_CLS = 20                  # training signals for the classifier's data
N_SCORE = 5                 # validation signals for the final score
THRESH_STEP = 0.01          # threshold sweep step

# noise seeds: one block per job, so no two jobs share a noise signal
TRAIN_SEED = 10_000         # detector epoch e uses TRAIN_SEED + e
CLS_SEED = 20_000           # classifier data signal s uses CLS_SEED + s
THRESH_SEED = 4000          # threshold signal s uses THRESH_SEED + s
SCORE_SEED = 5000           # scoring signal s uses SCORE_SEED + s

OUT_DIR = "Filter_compare/data"

# winners of tune_filters.py (data/tuning.json)
COMPETITORS = {
    "unfiltered": lambda x: x,
    "bandpass": lambda x: bandpass(x, low=100, high=1000, order=3),
    "wiener": lambda x: wiener(x, LEVEL, strength=5.66),
    "wavelet": lambda x: wavelet_denoise(x, "sym8", levels=3, rule="bayes", mode="soft"),
}


def noisy(d, seed, part):
    """D1_train + level noise with a fixed seed, then the training or validation part."""
    signal = d + generate_noise(LEVEL, len(d), seed=seed)
    return signal[:SPLIT] if part == "train" else signal[SPLIT:]


def train_detector(filt, d, train_index):
    """Step 1: the detector, trained on filtered noisy signals."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    n_windows = SPLIT - WIDTH + 1
    y = torch.tensor(make_labels(train_index, n_windows))

    model = make_detector()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(POS_WEIGHT)))
    start = time.time()
    for epoch in range(1, DET_EPOCHS + 1):
        x = windows(filt(noisy(d, TRAIN_SEED + epoch, "train")))
        order = torch.tensor(rng.permutation(n_windows))
        total = 0.0
        for i in range(0, n_windows, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(x[b]).squeeze(1), y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if epoch % 10 == 0:
            print(f"  detector epoch {epoch:3d}/{DET_EPOCHS}  "
                  f"{(time.time() - start) / 60:4.1f} min  loss {total / n_windows:.4f}",
                  flush=True)
    model.eval()
    return model


def choose_threshold(model, filt, d, val_index):
    """Step 2: the threshold with the best detection F, averaged over N_THRESH signals."""
    curves = []
    for s in range(N_THRESH):
        outputs = predict(model, windows(filt(noisy(d, THRESH_SEED + s, "val"))))
        res = sweep(outputs, val_index, step=THRESH_STEP)
        curves.append(res["f"])
    mean_f = np.mean(curves, axis=0)
    best = int(np.argmax(mean_f))
    return float(res["thresholds"][best]), float(mean_f[best])


def classifier_data(detector, threshold, filt, d, train_index, train_class):
    """Step 3: detections on N_CLS filtered training signals, with labels."""
    xs, ys = [], []
    for s in range(N_CLS):
        x, y, _, _, _ = detect_and_label(detector, threshold,
                                         filt(noisy(d, CLS_SEED + s, "train")),
                                         train_index, train_class)
        xs.append(x)
        ys.append(y)
    return torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(ys))


def train_classifier(x, y):
    """Step 4: the classifier, a fixed CLS_EPOCHS epochs."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = make_classifier()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, CLS_EPOCHS + 1):
        order = torch.tensor(rng.permutation(len(y)))
        for i in range(0, len(y), BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(x[b]), y[b]).backward()
            opt.step()
    model.eval()
    return model


def score(detector, threshold, classifier, filt, d, val_index, val_class):
    """Step 5: the whole pipeline on N_SCORE validation signals, graded."""
    rows = []
    for s in range(N_SCORE):
        x, _, guesses, _, _ = detect_and_label(detector, threshold,
                                               filt(noisy(d, SCORE_SEED + s, "val")),
                                               val_index, val_class)
        pred = classify(classifier, x)
        keep = pred != 0                       # drop detections called "fail"
        rows.append(grade(val_index, val_class, guesses[keep], pred[keep]))
    out = {}
    for k in ("f_total", "f_detect", "f_class"):
        values = [r[k] for r in rows]
        out[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    out["f_per_class"] = np.mean([r["f_per_class"] for r in rows], axis=0).round(3).tolist()
    return out


def run(name, d, index, cls):
    filt = COMPETITORS[name]
    folder = f"{OUT_DIR}/{name}"
    os.makedirs(folder, exist_ok=True)
    train = index < SPLIT
    train_index, train_class = index[train], cls[train]
    val_index, val_class = index[~train] - SPLIT, cls[~train]
    print(f"\n=== {name} ===", flush=True)
    start = time.time()

    detector = train_detector(filt, d, train_index)
    torch.save(detector.state_dict(), f"{folder}/detector.pt")
    det_minutes = (time.time() - start) / 60

    threshold, detect_f = choose_threshold(detector, filt, d, val_index)
    print(f"  threshold {threshold:.2f} (detection F {detect_f:.3f} on the threshold signals)",
          flush=True)

    x, y = classifier_data(detector, threshold, filt, d, train_index, train_class)
    print(f"  classifier data: {len(y)} detections, {(y == 0).sum().item()} fail", flush=True)
    classifier = train_classifier(x, y)
    torch.save(classifier.state_dict(), f"{folder}/classifier.pt")

    result = {"competitor": name, "level": LEVEL, "threshold": threshold,
              "threshold_detect_f": detect_f, "classifier_examples": len(y),
              "detector_minutes": round(det_minutes, 1),
              "total_minutes": round((time.time() - start) / 60, 1),
              **score(detector, threshold, classifier, filt, d, val_index, val_class)}
    with open(f"{folder}/result.json", "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"  F_total {result['f_total']['mean']:.3f} ±{result['f_total']['std']:.3f}  "
          f"({result['total_minutes']} min)", flush=True)


def show_table():
    """Print every competitor that has a result.json."""
    print(f"\nLevel {LEVEL}, {N_SCORE} validation signals, mean ± spread")
    print(f"{'competitor':12s} {'F_total':>13s} {'F_detect':>13s} {'F_class':>13s}   "
          f"class 1-5 F")
    for name in COMPETITORS:
        path = f"{OUT_DIR}/{name}/result.json"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            r = json.load(fh)
        cols = " ".join(f"{r[k]['mean']:.3f} ±{r[k]['std']:.3f}"
                        for k in ("f_total", "f_detect", "f_class"))
        print(f"{name:12s} {cols}   " + " ".join(f"{v:.2f}" for v in r["f_per_class"]))


def main():
    names = sys.argv[1:] or list(COMPETITORS)
    for name in names:
        if name not in COMPETITORS:
            raise SystemExit(f"unknown competitor {name!r}, choose from {list(COMPETITORS)}")
    data = np.load("Data/D1_train.npz")
    d, index, cls = data["d"], data["Index"], data["Class"]
    for name in names:
        run(name, d, index, cls)
    show_table()


if __name__ == "__main__":
    main()
