import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# load in the data
D1 = loadmat("Data/D1.mat")
d = D1["d"].ravel()
Index = D1["Index"].ravel()
Class = D1["Class"].ravel()

def show_graph():
    plt.plot(d[:1000])
    plt.xlabel("Sample")
    plt.ylabel("amplitude")
    plt.show()

show_graph()
