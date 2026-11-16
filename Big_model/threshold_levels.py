"""Find the best detection threshold for the big model (Big_model/best.pt,
epoch 2600, trained on level 6) on other noise levels.

For each level: add fresh noise to the validation part of D1_train
N_GRAPHS times, run the detector + peak picking, sweep the threshold from 0
to 1, and average the F, precision and recall curves.

Run from the project root (levels optional, default 4 5):
    python Big_model/threshold_levels.py 4 5

Saves:
  Big_model/data/thresholds.npz       averaged curves and the best threshold per level
  Artifacts/Big_Model_Threshold.png   chart: F, precision, recall vs threshold
"""
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, "Big_model")
sys.path.insert(0, "Pos_weight")
sys.path.insert(0, "Modular")
sys.path.insert(0, "Other")
from train_big import make_model                    # noqa: E402
from Test_POS_WEIGHT import windows, predict, SPLIT  # noqa: E402
from generate_noise import generate_noise           # noqa: E402
from find_threshold import sweep, LEVEL_COLOURS     # noqa: E402

MODEL = "Big_model/best.pt"
N_GRAPHS = 20             # fresh-noise signals per level
STEP = 0.005              # threshold step
OUT_DATA = "Big_model/data/thresholds.npz"
OUT_CHART = "Artifacts/Big_Model_Threshold.png"


def level_curves(model, level, d, val_index):
    """Average F, precision and recall curves over N_GRAPHS noisy signals."""
    f, p, r = [], [], []
    for g in range(N_GRAPHS):
        t0 = time.time()
        noisy = d[SPLIT:] + generate_noise(level, len(d))[SPLIT:]
        res = sweep(predict(model, windows(noisy)), val_index, step=STEP)
        f.append(res["f"]), p.append(res["precision"]), r.append(res["recall"])
        print(f"  level {level}, graph {g + 1}/{N_GRAPHS}: best F {res['best_f']:.3f} "
              f"at {res['best_threshold']:.3f}  ({time.time() - t0:.1f} s)", flush=True)
    f, p, r = np.array(f), np.array(p), np.array(r)
    mean_f = f.mean(0)
    best = int(np.argmax(mean_f))
    return {"thresholds": res["thresholds"], "mean_f": mean_f, "std_f": f.std(0),
            "precision": p.mean(0), "recall": r.mean(0), "best": best}


def show(curves):
    """Three panels (F, precision, recall), one line per level, dot = best F."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    for level, c in curves.items():
        colour, x, b = LEVEL_COLOURS[level - 1], c["thresholds"], c["best"]
        axes[0].fill_between(x, c["mean_f"] - c["std_f"], c["mean_f"] + c["std_f"],
                             color=colour, alpha=0.15, linewidth=0)
        axes[0].plot(x, c["mean_f"], color=colour, linewidth=2,
                     label=f"level {level}: best F {c['mean_f'][b]:.3f} at {x[b]:.3f}")
        axes[1].plot(x, c["precision"], color=colour, linewidth=2, label=f"level {level}")
        axes[2].plot(x, c["recall"], color=colour, linewidth=2, label=f"level {level}")
        for ax, y in zip(axes, (c["mean_f"], c["precision"], c["recall"])):
            ax.axvline(x[b], color=colour, linewidth=1, linestyle="--")
            ax.plot(x[b], y[b], "o", color=colour, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5)

    for ax, name in zip(axes, ["F-score (mean ± 1 sd)", "Precision", "Recall"]):
        ax.set_ylabel(name)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower center")
    axes[2].set_xlabel("Threshold (average output of the 9-point group)")
    fig.suptitle(f"Big model (epoch 2600, trained on level 6): detection vs threshold, "
                 f"averaged over {N_GRAPHS} fresh-noise graphs, ±50 samples")
    fig.tight_layout()
    return fig


def main():
    levels = [int(a) for a in sys.argv[1:]] or [4, 5]
    model = make_model()
    model.load_state_dict(torch.load(MODEL))
    model.eval()

    data = np.load("Data/D1_train.npz")
    d, index = data["d"], data["Index"]
    val_index = index[index >= SPLIT] - SPLIT

    curves, save = {}, {}
    for level in levels:
        c = curves[level] = level_curves(model, level, d, val_index)
        b = c["best"]
        print(f"Level {level}: best threshold {c['thresholds'][b]:.3f}  "
              f"F {c['mean_f'][b]:.3f} ± {c['std_f'][b]:.3f}  "
              f"precision {c['precision'][b]:.3f}  recall {c['recall'][b]:.3f}", flush=True)
        for k in ("mean_f", "std_f", "precision", "recall"):
            save[f"level{level}_{k}"] = c[k]
        save[f"level{level}_best_threshold"] = c["thresholds"][b]
    save["thresholds"] = c["thresholds"]

    np.savez_compressed(OUT_DATA, **save)
    show(curves).savefig(OUT_CHART, dpi=120)
    print(f"Saved {OUT_DATA} and {OUT_CHART}")


if __name__ == "__main__":
    main()
