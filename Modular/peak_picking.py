"""Turn a detector's output vector into a list of spike guesses.

Usage from another script (run from the project root):
    sys.path.insert(0, "Modular")
    from peak_picking import pick_peaks
    guesses = pick_peaks(outputs, width=9, threshold=0.5)
"""
import numpy as np


def pick_peaks(outputs, width=9, point=None, threshold=0.5):
    """Greedy peak picking.

    outputs:   1D array, the detector's score (0 to 1) for each window start.
    width:     how many adjacent scores are grouped together (e.g. 9).
    point:     which position in the group becomes the guess, counted from 1
               (e.g. 5 = the middle of 9). Default: the middle.
    threshold: stop when no group's average score is above this (0 to 1).

    Repeats: find the group of `width` adjacent scores with the biggest sum,
    put a guess at its `point`, set those scores to 0, then go again.
    Returns the guesses as a sorted array of 0-based positions in `outputs`.
    """
    if point is None:
        point = (width + 1) // 2
    if not 1 <= point <= width:
        raise ValueError(f"point must be between 1 and width ({width}), got {point}")

    scores = np.array(outputs, dtype=np.float64)   # a copy, so the input isn't changed
    kernel = np.ones(width)
    sums = np.convolve(scores, kernel, mode="valid")  # sums[i] = scores[i:i+width].sum()
    limit = threshold * width                         # average > threshold  <=>  sum > limit

    guesses = []
    while True:
        i = int(np.argmax(sums))           # start of the best group
        if sums[i] <= limit:
            break
        guesses.append(i + point - 1)

        scores[i:i + width] = 0            # remove the group's scores

        # only the sums that overlapped those scores change: starts i-width+1 to i+width-1
        lo = max(0, i - width + 1)
        hi = min(len(sums), i + width)
        sums[lo:hi] = np.convolve(scores[lo:hi + width - 1], kernel, mode="valid")

    return np.sort(np.array(guesses, dtype=np.int64))
