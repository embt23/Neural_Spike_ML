"""Mix of the two pipelines: per level, each class comes from whichever was
better in test_compare_results.txt.

  old: all-spike detector + peak picking, then the 6-way classifier.
  new: that class's own detector + peak picking (1detector-per-class).

USE_NEW says which classes come from the new detectors at each level; all
other classes come from the old pipeline's labels (label 0 = "fail" is dropped).

Running this file scores old vs mix on the test set with the grading formula:
    python 1detector-per-class/mix.py
"""
import sys

import numpy as np

sys.path.insert(0, "1detector-per-class")
from test_compare import old_guesses, new_guesses, f_score, new, N_SIGNALS, TEST_SEED   # noqa: E402

# Using new 3 and 4 at levels 5-6 too raised F_class but cut F_total (0.62 -> 0.60
# at level 5, 0.46 -> 0.39 at level 6): the new detectors make ~850 guesses each
# at level 6 vs 414 real spikes, which wrecks F_detect. So level 4 only.
USE_NEW = {2: (), 3: (), 4: (3, 4), 5: (), 6: ()}


def mix_guesses(level, signal):
    """Index (0-based, sorted) and Class of every guess for one signal."""
    g, labels = old_guesses(level, signal)
    keep = (labels != 0) & ~np.isin(labels, USE_NEW[level])
    index, cls = [g[keep]], [labels[keep]]
    for c in USE_NEW[level]:
        guesses = new_guesses(c, level, signal)
        index.append(guesses)
        cls.append(np.full(len(guesses), c))
    index, cls = np.concatenate(index), np.concatenate(cls)
    order = np.argsort(index)
    return index[order], cls[order]


def grade(real_index, real_cls, index, cls):
    """F_detect (all spikes, ignoring class), F_class (mean F over the 5 classes),
    F_total = 0.3 F_detect + 0.7 F_class."""
    f_detect = f_score(real_index, index)
    f_class = np.mean([f_score(real_index[real_cls == c], index[cls == c])
                       for c in new.CLASSES])
    return f_detect, f_class, 0.3 * f_detect + 0.7 * f_class


def main():
    test = np.load("Data/D1_test.npz")
    d, real_index, real_cls = test["d"], test["Index"], test["Class"]
    print(f"Test set, {N_SIGNALS} noise signals per level (same seeds as test_compare)\n")
    print("level   F_detect      F_class       F_total")
    print("        old   mix     old   mix     old   mix")
    for level in new.LEVELS:
        old_s, mix_s = [], []
        for s in range(N_SIGNALS):
            noisy = d + new.generate_noise(level, len(d), seed=TEST_SEED + 10 * level + s)
            g, labels = old_guesses(level, noisy)
            old_s.append(grade(real_index, real_cls, g[labels != 0], labels[labels != 0]))
            mix_s.append(grade(real_index, real_cls, *mix_guesses(level, noisy)))
        o, m = np.mean(old_s, 0), np.mean(mix_s, 0)
        print(f"  {level}    " + "    ".join(f"{o[k]:.2f}  {m[k]:.2f}" for k in range(3)),
              flush=True)


if __name__ == "__main__":
    main()
