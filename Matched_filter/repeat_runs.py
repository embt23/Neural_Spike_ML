"""Repeat the matched filter + peeling N_RUNS times per level, with brand-new
noise every run, and estimate the final grade.

Each run, for each level:
  - new noise for the covariance (the filters are rebuilt from it)
  - new noise added to the validation signal D1_train[SPLIT:]
  - threshold: the one compare.py chose (Matched_filter/data/thresholds.json)
Scored with F_detect, F_class (mean over the 5 classes) and
F_total = 0.3 F_detect + 0.7 F_class. Then:
    grade = 0.1 L2 + 0.15 L3 + 0.2 L4 + 0.25 L5 + 0.3 L6
Validation data only; the test set is not touched.

Run from the project root (runs optional, default 10):
    python Matched_filter/repeat_runs.py 10
Saves Matched_filter/data/repeat_runs.json (every run's scores and seeds).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "Matched_filter")
from matched_filter import MatchedFilterBank, templates, noise_covariance, SPLIT  # noqa: E402
from generate_noise import generate_noise                                       # noqa: E402
from compare import grade                                                        # noqa: E402

WEIGHTS = {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.30}
OUT = "Matched_filter/data/repeat_runs.json"


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t = np.load("Data/D1_train.npz")
    d = t["d"]
    val = t["Index"] >= SPLIT
    real_i, real_c = t["Index"][val] - SPLIT, t["Class"][val]
    with open("Matched_filter/data/thresholds.json") as fh:
        thresholds = {int(k): v for k, v in json.load(fh).items()}
    T = templates()

    runs = {level: [] for level in WEIGHTS}
    rng = np.random.default_rng(time.time_ns())        # different seeds every time it's run
    for level in WEIGHTS:
        for r in range(n_runs):
            cov_seed, sig_seed = (int(s) for s in rng.integers(0, 2**31, size=2))
            bank = MatchedFilterBank(T, noise_covariance(level, seed=cov_seed))
            x = d[SPLIT:] + generate_noise(level, len(d), seed=sig_seed)[SPLIT:]
            fd, fc, ft = grade(real_i, real_c, *bank.peel(x, thresholds[level]))
            runs[level].append({"f_detect": fd, "f_class": fc, "f_total": ft,
                                "cov_seed": cov_seed, "signal_seed": sig_seed})
            print(f"level {level}  run {r + 1:2d}  F_detect {fd:.3f}  F_class {fc:.3f}  "
                  f"F_total {ft:.3f}", flush=True)

    print("\nlevel  F_total mean ± std  (min to max)   weight")
    grade_runs = np.zeros(n_runs)
    for level, w in WEIGHTS.items():
        ft = np.array([row["f_total"] for row in runs[level]])
        grade_runs += w * ft
        print(f"  {level}    {ft.mean():.3f} ± {ft.std():.3f}     ({ft.min():.3f} to {ft.max():.3f})"
              f"   {w:.2f}")
    print(f"\nEstimated grade: {grade_runs.mean():.3f} ± {grade_runs.std():.3f} "
          f"(min {grade_runs.min():.3f}, max {grade_runs.max():.3f} over {n_runs} runs)")

    with open(OUT, "w") as fh:
        json.dump({"runs": runs, "grade_per_run": grade_runs.round(4).tolist()}, fh, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
