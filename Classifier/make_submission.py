"""Predict Index and Class for D2 to D6 and save them as .mat files.

For each dataset: run that level's detector + peak picking (settings from
make_dataset.py), cut the 80-sample window around every detection, classify
it with Classifier/level<L>.pt, drop the ones called "fail" (0), and save
Index (1-based, sorted) and Class like D1.mat stores them (no d).

Run from the project root (datasets optional, default 2 3 4 5 6):
    python Classifier/make_submission.py 2 3 4 5 6

Saves Submissions/<today>/D<n>.mat.
"""
import datetime
import os
import sys

import numpy as np
import torch
from scipy.io import loadmat, savemat

sys.path.insert(0, "Classifier")
from make_dataset import DETECTORS, BEFORE, AFTER, small, pick_peaks   # noqa: E402
from train_classifier import make_model as make_classifier, classify   # noqa: E402

OUT_DIR = f"Submissions/{datetime.date.today()}"


def predict_dataset(n):
    d = loadmat(f"Data/D{n}.mat")["d"].ravel()
    make_model, weights, threshold, _, _ = DETECTORS[n]
    detector = make_model()
    detector.load_state_dict(torch.load(weights))
    detector.eval()
    classifier = make_classifier()
    classifier.load_state_dict(torch.load(f"Classifier/level{n}.pt"))
    classifier.eval()

    guesses = pick_peaks(small.predict(detector, small.windows(d)), threshold=threshold)
    guesses = guesses[(guesses >= BEFORE) & (guesses + AFTER < len(d))]
    x = np.stack([d[p - BEFORE:p + AFTER + 1] for p in guesses]).astype(np.float32)
    labels = classify(classifier, x)

    keep = labels != 0
    index, cls = guesses[keep] + 1, labels[keep]           # +1: back to 1-based (MATLAB)
    counts = np.bincount(cls, minlength=6)[1:]
    print(f"D{n}: {len(guesses)} detections, {(~keep).sum()} called fail, "
          f"{len(index)} submitted  classes 1-5: {counts}", flush=True)
    return index, cls


def main():
    datasets = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6]
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in datasets:
        index, cls = predict_dataset(n)
        savemat(f"{OUT_DIR}/D{n}.mat", {"Index": index.astype(np.int32)[None, :],
                                        "Class": cls.astype(np.uint8)[None, :]})
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
