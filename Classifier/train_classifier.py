"""Train one classifier per level on the detections from make_dataset.py.

Network: 80 inputs (9 before the detected point, the point, 70 after)
-> 100 -> 80 -> 40 -> 6 outputs (0 = fail, 1-5 = class), ReLU between.
Loss: cross-entropy (no class weights). Adam, lr 0.001, batch 256.

Every CHECK_EVERY epochs the model is scored on the validation signals with
the full grader (V0/grading.py, F_total = 0.3 F_detect + 0.7 F_class):
detections it calls "fail" are dropped, the rest keep the predicted class.
The best-scoring epoch is saved.

Run from the project root (levels optional, default 4 5 6):
    python Classifier/train_classifier.py 4 5 6

Saves Classifier/level<L>.pt (best epoch) and Classifier/data/log_level<L>.jsonl.
"""
import json
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "V0")
from grading import grade   # noqa: E402

LAYERS = [80, 100, 80, 40, 6]
EPOCHS = 300
CHECK_EVERY = 5
BATCH = 256
LR = 0.001
SEED = 0
SPLIT = 1_000_000


def make_model():
    layers = []
    for a, b in zip(LAYERS[:-1], LAYERS[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])     # no ReLU after the output layer


def classify(model, x):
    """Predicted label 0-5 for each window."""
    with torch.no_grad():
        return model(torch.as_tensor(x)).argmax(1).numpy()


def score(pred, data, true_index, true_class):
    """Grader score averaged over the validation signals."""
    rows = []
    for s in np.unique(data["signal_val"]):
        keep = (data["signal_val"] == s) & (pred != 0)
        pos, cls = data["pos_val"][keep], pred[keep]
        order = np.argsort(pos)
        rows.append(grade(true_index, true_class, pos[order], cls[order]))
    return {k: float(np.mean([r[k] for r in rows])) for k in ("f_total", "f_detect", "f_class")}


def train_level(level, true_index, true_class):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data = np.load(f"Classifier/data/level{level}.npz")
    x, y = torch.tensor(data["x_train"]), torch.tensor(data["y_train"])

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    log = open(f"Classifier/data/log_level{level}.jsonl", "w")
    best, start = None, time.time()

    for epoch in range(1, EPOCHS + 1):
        order = torch.tensor(rng.permutation(len(y)))
        total = 0.0
        for i in range(0, len(y), BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(x[b]), y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)

        if epoch % CHECK_EVERY == 0:
            pred = classify(model, data["x_val"])
            row = {"epoch": epoch, "minutes": round((time.time() - start) / 60, 2),
                   "loss": total / len(y),
                   "val_accuracy": float((pred == data["y_val"]).mean()),
                   **score(pred, data, true_index, true_class)}
            log.write(json.dumps(row) + "\n")
            if best is None or row["f_total"] > best["f_total"]:
                best = row
                torch.save(model.state_dict(), f"Classifier/level{level}.pt")
            if epoch % 50 == 0:
                print(f"  level {level} epoch {epoch}: loss {row['loss']:.3f}  "
                      f"val acc {row['val_accuracy']:.3f}  F_total {row['f_total']:.3f}",
                      flush=True)
    log.close()
    return best


def main():
    levels = [int(a) for a in sys.argv[1:]] or [4, 5, 6]
    d1 = np.load("Data/D1_train.npz")
    val = d1["Index"] >= SPLIT
    true_index, true_class = d1["Index"][val] - SPLIT, d1["Class"][val]

    for level in levels:
        data = np.load(f"Classifier/data/level{level}.npz")
        # baseline: keep every detection and give it its true class if matched
        # (the best a classifier could do without rejecting fails)
        base = score(np.where(data["y_val"] == 0, 1, data["y_val"]), data,
                     true_index, true_class)
        best = train_level(level, true_index, true_class)
        print(f"Level {level}: best epoch {best['epoch']}  F_total {best['f_total']:.3f}  "
              f"(F_detect {best['f_detect']:.3f}, F_class {best['f_class']:.3f}, "
              f"val acc {best['val_accuracy']:.3f})", flush=True)
        print(f"  for reference, perfect classes but no fail rejection: "
              f"F_total {base['f_total']:.3f}", flush=True)


if __name__ == "__main__":
    main()
