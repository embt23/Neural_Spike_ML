"""Classify the detection errors of one model on the fresh-noise graphs.

Run from the project root:
    python Pos_weight/error_analysis.py
Prints a report and saves example screenshots of each error group to
Artifacts/Pos_Weight/errors/.

Error groups
  Missed spikes (real spike with no guess within ±50 samples), by how strongly
  the detector responded near the spike's start:
    buried   best 9-point average < 0.1: the detector didn't see it at all
    weak     0.1 to THRESHOLD: it saw something, but below the threshold
    strong   >= THRESHOLD but still missed: the guess went to a neighbour or
             landed more than ±50 away
  False guesses (guess with no real spike paired to it), by distance to the
  nearest real spike:
    duplicate  <= 50 samples: a second guess on a spike that was already found
    near       51 to 100 samples: close to a spike but outside the tolerance
    noise      > 100 samples: nothing there, the noise fooled the detector
"""
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "Modular")
from peak_picking import pick_peaks  # noqa: E402
from f_score import f_detect         # noqa: E402

LEVEL = 6
POS_WEIGHT = 5
THRESHOLD = 0.5
WIDTH = 9
TOLERANCE = 50
FS = 25000
SPLIT = 1_000_000
DATA_DIR = "Pos_weight/data/fresh_noise"
OUT_DIR = "Artifacts/Pos_Weight/errors"
N_EXAMPLES = 3           # screenshots per error group
SEED = 0

MS1, MS2 = 25, 50        # 1 ms and 2 ms in samples

CLASS_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}
SIGNAL, CLEAN, OUTPUT, HIT, FALSE = "#9db8de", "#333333", "#eb6834", "#008300", "#d03030"


def nearest_other(positions, others, exclude_self=False):
    """Distance from each position to the nearest of `others`."""
    d = np.abs(positions[:, None] - others[None, :])
    if exclude_self:
        d[d == 0] = 10**9
    return d.min(axis=1) if len(others) else np.full(len(positions), 10**9)


def analyse():
    files = sorted(glob.glob(f"{DATA_DIR}/level{LEVEL}_graph*.npz"))
    first = np.load(files[0])
    real, cls = first["val_index"], first["val_class"]
    row = list(first["pos_weights"]).index(POS_WEIGHT)
    gap = nearest_other(real, real, exclude_self=True)   # distance to the nearest other real spike

    misses, falses = [], []      # one dict per error
    n_guesses = n_hits = 0
    for gi, f in enumerate(files):
        r = np.load(f)
        out = r["outputs"][row].astype(np.float64)
        guesses, scores = pick_peaks(out, WIDTH, threshold=THRESHOLD, return_scores=True)
        res = f_detect(real, guesses, TOLERANCE)
        n_guesses += len(guesses)
        n_hits += res["hits"]
        avg = np.convolve(out, np.ones(WIDTH) / WIDTH, mode="valid")  # 9-point averages

        missed = np.ones(len(real), bool)
        missed[res["real_matched"]] = False
        for k in np.where(missed)[0]:
            s = real[k]
            response = avg[max(0, s - 14):s + 6].max()   # groups centred within ±10 of the start
            group = "buried" if response < 0.1 else "weak" if response < THRESHOLD else "strong"
            misses.append({"graph": gi, "file": f, "pos": s, "cls": cls[k], "gap": gap[k],
                           "response": response, "group": group})

        false = np.ones(len(guesses), bool)
        false[res["guess_matched"]] = False
        dist = nearest_other(guesses, real)
        to_guess = nearest_other(guesses, guesses, exclude_self=True)
        for k in np.where(false)[0]:
            group = "duplicate" if dist[k] <= 50 else "near" if dist[k] <= 100 else "noise"
            falses.append({"graph": gi, "file": f, "pos": guesses[k], "dist": dist[k],
                           "to_guess": to_guess[k], "score": scores[k], "group": group})

    return files, real, cls, gap, misses, falses, n_guesses, n_hits


def report(files, real, cls, gap, misses, falses, n_guesses, n_hits):
    n = len(files)
    total = n * len(real)
    print(f"Level {LEVEL}, pw{POS_WEIGHT}, threshold {THRESHOLD}, {n} graphs, "
          f"{len(real)} real spikes per graph ({total} in total)")
    p, r = n_hits / n_guesses, n_hits / total
    print(f"  guesses {n_guesses}, hits {n_hits}: precision {p:.3f}, recall {r:.3f}, "
          f"F {2 * p * r / (p + r):.3f}")

    # ---- 1. missed spikes ----
    m_cls = np.array([m["cls"] for m in misses])
    m_gap = np.array([m["gap"] for m in misses])
    print(f"\n1. MISSED SPIKES: {len(misses)} of {total} ({len(misses) / total:.1%}) "
          f"-> recall {1 - len(misses) / total:.3f}")
    print("   class  spikes  missed  missed %  | of the missed: neighbour <=1 ms  <=2 ms"
          "  | all spikes of class: <=1 ms  <=2 ms")
    for c in range(1, 6):
        here = m_cls == c
        n_c = n * (cls == c).sum()
        print(f"   {c:>5}  {n_c:>6}  {here.sum():>6}  {here.sum() / n_c:>7.1%}  |"
              f"  {np.mean(m_gap[here] <= MS1):>27.1%}  {np.mean(m_gap[here] <= MS2):>6.1%}"
              f"  |  {np.mean(gap[cls == c] <= MS1):>26.1%}  {np.mean(gap[cls == c] <= MS2):>6.1%}")
    print(f"   all    {total:>6}  {len(misses):>6}  {len(misses) / total:>7.1%}  |"
          f"  {np.mean(m_gap <= MS1):>27.1%}  {np.mean(m_gap <= MS2):>6.1%}"
          f"  |  {np.mean(gap <= MS1):>26.1%}  {np.mean(gap <= MS2):>6.1%}")
    print("   (the right-hand columns are the normal rate, to compare against)")

    print("\n   Missed spikes by how the detector responded:")
    for g, text in [("buried", "< 0.1, didn't see it"),
                    ("weak", f"0.1-{THRESHOLD}, saw it but below threshold"),
                    ("strong", f">= {THRESHOLD}, but the guess went elsewhere")]:
        sel = [m for m in misses if m["group"] == g]
        counts = np.bincount([m["cls"] for m in sel], minlength=6)[1:]
        print(f"   {g:<7} {len(sel):>5} ({len(sel) / len(misses):.0%})  {text:<42}"
              f" by class 1-5: {counts.tolist()}")

    times = np.array([sum(1 for m in misses if m["pos"] == s) for s in real])
    print(f"\n   Per real spike over the {n} graphs: never missed {np.sum(times == 0)}, "
          f"always missed {np.sum(times == n)}, sometimes {np.sum((times > 0) & (times < n))}")

    # ---- 2. false guesses ----
    f_dist = np.array([f["dist"] for f in falses])
    print(f"\n2. FALSE GUESSES (hallucinated): {len(falses)} of {n_guesses} guesses "
          f"({len(falses) / n_guesses:.1%}) -> precision {1 - len(falses) / n_guesses:.3f}")
    print(f"   within 1 ms of a real spike: {np.mean(f_dist <= MS1):.1%}, "
          f"within 2 ms: {np.mean(f_dist <= MS2):.1%}")
    for g, text in [("duplicate", "<= 50 samples: second guess on a found spike"),
                    ("near", "51-100 samples from a spike"),
                    ("noise", "> 100 samples: nothing there")]:
        sel = [f for f in falses if f["group"] == g]
        sc = np.array([f["score"] for f in sel]) if sel else np.array([np.nan])
        print(f"   {g:<9} {len(sel):>5} ({len(sel) / len(falses):.0%})  {text:<46}"
              f" median score {np.median(sc):.2f}")

    dup = np.array([f["to_guess"] for f in falses if f["group"] == "duplicate"])
    if len(dup):
        print(f"\n   Duplicates: distance to the nearest other guess: median {np.median(dup):.0f} "
              f"samples, {np.mean(dup <= 2 * WIDTH):.0%} within {2 * WIDTH} samples")


def screenshot(err, kind, real, cls, path, half=150):
    """Plot ±half samples around one error: signal, clean D1, output, spikes, guesses."""
    r = np.load(err["file"])
    row = list(r["pos_weights"]).index(POS_WEIGHT)
    out = r["outputs"][row].astype(np.float64)
    noisy = r["signal"]
    clean = np.load("Data/D1_train.npz")["d"][SPLIT:]
    lo, hi = max(0, err["pos"] - half), min(len(out), err["pos"] + half)
    t = np.arange(lo, hi)

    guesses = pick_peaks(out, WIDTH, threshold=THRESHOLD)
    res = f_detect(real, guesses, TOLERANCE)
    hit = np.zeros(len(guesses), bool)
    hit[res["guess_matched"]] = True

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), height_ratios=[2, 1])
    top.plot(t, noisy[lo:hi], color=SIGNAL, linewidth=0.8, label="signal (level 6 noise)")
    top.plot(t, clean[lo:hi], color=CLEAN, linewidth=1, label="clean D1")
    bottom.plot(t, out[lo:hi], color=OUTPUT, linewidth=1.2, label="output")
    bottom.axhline(THRESHOLD, color="#999999", linestyle="--", linewidth=1)

    for s, c in zip(real, cls):
        if lo <= s < hi:
            for ax in (top, bottom):
                ax.axvline(s, color=CLASS_COLOURS[c], linewidth=2, alpha=0.8, zorder=0)
            top.annotate(f"class {c}", (s, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(3, -12), textcoords="offset points", fontsize=9)
    for g, h in zip(guesses, hit):
        if lo <= g < hi:
            bottom.plot(g, 1.08, "v", color=HIT if h else FALSE, markersize=9)

    top.legend(frameon=False, loc="lower right", fontsize=9)
    bottom.set_ylim(-0.05, 1.15)
    bottom.set_ylabel("Output")
    top.set_ylabel("Signal")
    bottom.set_xlabel("Sample in validation signal")
    for ax in (top, bottom):
        ax.spines[["top", "right"]].set_visible(False)
    extra = (f"class {err['cls']}, response {err['response']:.2f}, nearest other spike "
             f"{err['gap']} samples" if kind == "missed" else
             f"score {err['score']:.2f}, nearest real spike {err['dist']} samples")
    fig.suptitle(f"{kind} - {err['group']} (graph {err['graph'] + 1}, sample {err['pos']}): {extra}\n"
                 f"coloured lines = real spikes by class, triangles = guesses "
                 f"(green hit, red false), dashed = threshold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_examples(real, cls, misses, falses):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for kind, errors, groups in [("missed", misses, ["buried", "weak", "strong"]),
                                 ("false", falses, ["duplicate", "near", "noise"])]:
        for g in groups:
            sel = [e for e in errors if e["group"] == g]
            for k, i in enumerate(rng.choice(len(sel), min(N_EXAMPLES, len(sel)), replace=False)):
                screenshot(sel[i], kind, real, cls, f"{OUT_DIR}/{kind}_{g}_{k + 1}.png")


if __name__ == "__main__":
    files, real, cls, gap, misses, falses, n_guesses, n_hits = analyse()
    report(files, real, cls, gap, misses, falses, n_guesses, n_hits)
    save_examples(real, cls, misses, falses)
    print(f"\nScreenshots saved to {OUT_DIR}/")
