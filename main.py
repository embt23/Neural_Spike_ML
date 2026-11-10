import numpy as np
import matplotlib.pyplot as plt

FS = 25000 # sampling Frequency

# load in the data
train = np.load("Data/D1_train.npz")
d = train["d"]
Index = train["Index"]
Class = train["Class"]
