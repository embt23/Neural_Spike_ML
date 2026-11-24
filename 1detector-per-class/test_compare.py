"""Old pipeline vs one-detector-per-class, on the test set (D1_test).

For each level, N_SIGNALS noisy versions of the test signal are made (fixed
seeds, different from the validation seeds), and each class is scored with
F against that class's test spikes (±50 samples):

  old: all-spike detector + peak picking at its fixed threshold, then the
       6-way classifier; the detections it labels as class C are C's guesses.
  new: class C's detector (best.pt) + peak picking, keeping peaks above the
       threshold that was best on the validation signal at that epoch.

Nothing is tuned on the test set: every threshold comes from validation.

Run from the project root:
    python 1detector-per-class/test_compare.py
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "1detector-per-class")
sys.path.insert(0, "Classifier")
import main as new                                                       # noqa: E402
from make_dataset import DETECTORS, BEFORE, AFTER, small                 # noqa: E402
from make_dataset import pick_peaks as old_pick_peaks                    # noqa: E402
from train_classifier import make_model as make_classifier, classify     # noqa: E402

N_SIGNALS = 5              # noisy versions of the test signal per level
TEST_SEED = 2000           # noise seed = TEST_SEED + 10 * level + signal number


def f_score(real, guesses):
    hits = new.count_hits(real, guesses)
    return 2 * hits / (len(real) + len(guesses)) if hits else 0.0   # = 2PR/(P+R)


def old_guesses(level, noisy):
    """Class label for every detection of the old pipeline."""
    make_model, weights, threshold, _, _ = DETECTORS[level]
    detector = make_model()
    detector.load_state_dict(torch.load(weights))
    detector.eval()
    classifier = make_classifier()
    classifier.load_state_dict(torch.load(f"Classifier/level{level}.pt"))
    classifier.eval()

    g = old_pick_peaks(small.predict(detector, small.windows(noisy)), threshold=threshold)
    g = g[(g >= BEFORE) & (g + AFTER < len(noisy))]
    labels = classify(classifier, np.stack([noisy[p - BEFORE:p + AFTER + 1]
                                            for p in g]).astype(np.float32))
    return g, labels


def new_guesses(target, level, noisy):
    """Guesses of class `target`'s detector at its validation threshold."""
    folder = f"{new.OUT_DIR}/class{target}_level{level}"
    log = [json.loads(line) for line in open(f"{folder}/log.jsonl")]
    threshold = max(log, key=lambda row: row["f"])["threshold"]   # the best.pt epoch
    model = new.make_model()
    model.load_state_dict(torch.load(f"{folder}/best.pt"))
    model.eval()
    guesses, scores = new.pick_peaks(new.predict(model, new.windows(noisy)))
    return guesses[scores > threshold]


def main():
    test = np.load("Data/D1_test.npz")
    d, index, cls = test["d"], test["Index"], test["Class"]
    print(f"Test set: {len(index)} spikes, classes 1-5: {np.bincount(cls)[1:]}, "
          f"{N_SIGNALS} noise signals per level\n")
    print("level  " + "  ".join(f"class {c}: old -> new" for c in new.CLASSES))

    for level in new.LEVELS:
        old_f = np.zeros((N_SIGNALS, 5))
        new_f = np.zeros((N_SIGNALS, 5))
        for s in range(N_SIGNALS):
            noisy = d + new.generate_noise(level, len(d), seed=TEST_SEED + 10 * level + s)
            g, labels = old_guesses(level, noisy)
            for k, c in enumerate(new.CLASSES):
                real = index[cls == c]
                old_f[s, k] = f_score(real, g[labels == c])
                new_f[s, k] = f_score(real, new_guesses(c, level, noisy))
        print(f"  {level}    " + "  ".join(
            f"{o:.2f} -> {n:.2f}      " for o, n in zip(old_f.mean(0), new_f.mean(0))),
            flush=True)


if __name__ == "__main__":
    main()
