"""Predict Index and Class for D2 to D6 with the mix (mix.py) and save .mat files.

Dataset Dn uses noise level n. Index is saved 1-based and sorted, Class paired
with it, like D1.mat (no d). Also zips the folder for submitting.

Run from the project root (datasets optional, default 2 3 4 5 6):
    python 1detector-per-class/make_submission.py

Saves Submissions/<today>/D<n>.mat and Submissions/<today>.zip.
"""
import datetime
import os
import shutil
import sys

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, "1detector-per-class")
from mix import mix_guesses, USE_NEW   # noqa: E402

OUT_DIR = f"Submissions/{datetime.date.today()}"


def main():
    datasets = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6]
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in datasets:
        d = loadmat(f"Data/D{n}.mat")["d"].ravel()
        index, cls = mix_guesses(n, d)
        counts = np.bincount(cls, minlength=6)[1:]
        print(f"D{n}: {len(index)} submitted, classes 1-5: {counts}, "
              f"new detectors for classes {USE_NEW[n] or 'none'}", flush=True)
        savemat(f"{OUT_DIR}/D{n}.mat", {"Index": (index + 1).astype(np.int32)[None, :],
                                        "Class": cls.astype(np.uint8)[None, :]})
    shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
    print(f"Saved to {OUT_DIR}/ and {OUT_DIR}.zip")


if __name__ == "__main__":
    main()
