"""Test the saved models on brand-new noise, many times, and average.

For each level, N_GRAPHS times: make a new noisy validation signal (clean D1
validation part + fresh noise), run every POS_WEIGHT model of a run over that
same signal and sweep the threshold from 0 to 1. Then average each model's
F, precision and recall at every threshold over the N_GRAPHS signals, and
find the peak (threshold, F) of the averaged F curve.

Run from the project root (run folder, then the levels; default 1-20, 1 to 6):
    python Pos_weight/fresh_noise_test.py 1-20 1 2 3 4 5 6

Saves to Pos_weight/data/fresh_noise/:
  level<L>_graph<GG>.npz  one per graph: the noisy signal, the real spikes and
                          classes, every model's output vector and its
                          F/precision/recall curves (for looking at errors later)
  summary.npz             the averaged curves and peaks
"""
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "Pos_weight")
sys.path.insert(0, "Other")
sys.path.insert(0, "Modular")
from Test_POS_WEIGHT import make_model, windows, predict, SPLIT  # noqa: E402
from generate_noise import generate_noise                          # noqa: E402
from find_threshold import sweep                                   # noqa: E402

DATA_DIR = "Pos_weight/data"
OUT_DIR = "Pos_weight/data/fresh_noise"
N_GRAPHS = 20      # new noisy signals per level (level 1 has no noise, so runs once)


def load_models(run, level):
    """{pos_weight: model} for every saved model of this run and level."""
    models = {}
    for path in glob.glob(f"{DATA_DIR}/{run}/level{level}_pw*.pt"):
        pw = int(path.split("_pw")[1].split(".")[0])
        models[pw] = make_model()
        models[pw].load_state_dict(torch.load(path))
    return dict(sorted(models.items()))


def test_level(run, level, clean_val, val_index, val_class, n_graphs=N_GRAPHS):
    """Sweep every model on n_graphs new signals, saving each graph's data.
    Returns {pos_weight: {"f": (n_graphs, n_thresholds), "precision": ..., "recall": ...}}."""
    models = load_models(run, level)
    if level == 1:
        n_graphs = 1   # no added noise: every signal would be identical
    curves = {pw: {"f": [], "precision": [], "recall": []} for pw in models}

    for g in range(n_graphs):
        start = time.time()
        noisy = clean_val
        if level > 1:
            noisy = clean_val + generate_noise(level, SPLIT + len(clean_val))[SPLIT:]
        x = windows(noisy)
        graph = {"signal": noisy.astype(np.float32), "val_index": val_index,
                 "val_class": val_class, "pos_weights": np.array(list(models)),
                 "level": level, "run": run}
        outputs = []
        for pw, model in models.items():
            out = predict(model, x)
            outputs.append(out)
            res = sweep(out, val_index)
            for key in curves[pw]:
                curves[pw][key].append(res[key])
                graph[f"pw{pw}_{key}"] = res[key]
        graph["outputs"] = np.stack(outputs)   # (n_models, n_windows), rows in pos_weights order
        graph["thresholds"] = res["thresholds"]
        np.savez_compressed(f"{OUT_DIR}/level{level}_graph{g + 1:02d}.npz", **graph)
        print(f"  level {level}, graph {g + 1}/{n_graphs}  ({time.time() - start:.1f} s)")

    thresholds = res["thresholds"]
    return thresholds, {pw: {k: np.array(v) for k, v in c.items()} for pw, c in curves.items()}


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "1-20"
    levels = [int(a) for a in sys.argv[2:]] or [1, 2, 3, 4, 5, 6]

    data = np.load("Data/D1_train.npz")
    clean_val = data["d"][SPLIT:]
    is_val = data["Index"] >= SPLIT
    val_index, val_class = data["Index"][is_val] - SPLIT, data["Class"][is_val]
    os.makedirs(OUT_DIR, exist_ok=True)

    saved, table = {}, []
    for level in levels:
        thresholds, curves = test_level(run, level, clean_val, val_index, val_class)
        saved["thresholds"] = thresholds
        for pw, c in curves.items():
            mean_f = c["f"].mean(axis=0)            # average F at each threshold
            b = int(np.argmax(mean_f))              # peak of the averaged curve
            peaks = c["f"].max(axis=1)              # each graph's own peak F
            table.append((level, pw, thresholds[b], mean_f[b], c["f"][:, b].std(),
                          c["precision"].mean(axis=0)[b], c["recall"].mean(axis=0)[b],
                          peaks.min(), peaks.max(), len(c["f"])))

            p = f"level{level}_pw{pw}_"
            saved[p + "mean_f"] = mean_f
            saved[p + "std_f"] = c["f"].std(axis=0)
            saved[p + "mean_precision"] = c["precision"].mean(axis=0)
            saved[p + "mean_recall"] = c["recall"].mean(axis=0)
            saved[p + "peak"] = np.array([thresholds[b], mean_f[b]])

    np.savez_compressed(f"{OUT_DIR}/summary.npz", n_graphs=N_GRAPHS, run=run, **saved)

    print(f"\nPeak of the averaged F curve ({N_GRAPHS} new graphs per level, level 1: 1)")
    print("  level  pos_weight  threshold  mean F  ±std   precision  recall"
          "  (graph peaks min-max)")
    for level, pw, t, f, sd, prec, rec, lo, hi, n in table:
        print(f"  {level:>5}  {pw:>10}  {t:>9.3f}  {f:.3f}  ±{sd:.3f}  {prec:>9.3f}  "
              f"{rec:.3f}   ({lo:.3f}-{hi:.3f})")
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
