"""Predict Index and Class for D2 to D6 with the matched filter bank + peeling.

Dataset Dn uses noise level n and the threshold compare.py chose for that
level on validation signals (Matched_filter/data/thresholds.json). Index is
saved 1-based and sorted, Class paired with it, like D1.mat (no d). Also zips
the folder for submitting.

Run from the project root (after compare.py; datasets optional, default 2-6):
    python Matched_filter/make_submission.py

Saves Submissions/<today>_matched_filter/D<n>.mat and the .zip.
"""
import datetime
import json
import os
import shutil
import sys

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, "Matched_filter")
from matched_filter import build_bank    # noqa: E402

OUT_DIR = f"Submissions/{datetime.date.today()}_matched_filter"


def main():
    datasets = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6]
    with open("Matched_filter/data/thresholds.json") as fh:
        thresholds = {int(k): v for k, v in json.load(fh).items()}
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in datasets:
        d = loadmat(f"Data/D{n}.mat")["d"].ravel()
        index, cls = build_bank(n).peel(d, thresholds[n])
        print(f"D{n}: {len(index)} spikes, classes 1-5: {np.bincount(cls, minlength=6)[1:]}",
              flush=True)
        savemat(f"{OUT_DIR}/D{n}.mat", {"Index": (index + 1).astype(np.int32)[None, :],
                                        "Class": cls.astype(np.uint8)[None, :]})
    shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
    print(f"Saved to {OUT_DIR}/ and {OUT_DIR}.zip")


if __name__ == "__main__":
    main()
