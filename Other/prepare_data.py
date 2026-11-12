import numpy as np
from scipy.io import loadmat

SPLIT = 1_200_000  # first 1.2M samples = training, rest = testing

# Load
D1 = loadmat("Data/D1.mat")
d = D1["d"].ravel()
Index = D1["Index"].ravel() - 1  # MATLAB counts from 1, Python from 0
Class = D1["Class"].ravel()

# Put spikes in time order, keeping each Class with its spike
order = np.argsort(Index)
Index = Index[order]
Class = Class[order]

# Split
is_train = Index < SPLIT

np.savez("Data/D1_train.npz",
         d=d[:SPLIT],
         Index=Index[is_train],
         Class=Class[is_train])

np.savez("Data/D1_test.npz",
         d=d[SPLIT:],
         Index=Index[~is_train] - SPLIT,  # shift so 0 = start of test section
         Class=Class[~is_train])

print(f"Train: {is_train.sum()} spikes, test: {(~is_train).sum()} spikes")
