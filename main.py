import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from grading import f_score


FS = 25000 # sampling Frequency
STEP = 10 # how many samples the window jumps each time
WIDTH = 75 # how many samples each window looks at
LEARNING_RATE = 0.001 # how big each update nudge is
EPOCHS = 20 # how many full passes through the training data
BATCH_SIZE = 256 # how many windows per update
THRESHOLD = 0.5 # output above this = spike


# load in the data
train = np.load("Data/D1_train.npz")
d = train["d"]
Index = train["Index"]
Class = train["Class"]


# Build windows and their targets
starts = np.arange(0, len(d) - WIDTH + 1, STEP)
labels = np.zeros((len(starts), 5), dtype=np.float32)
spike_window = Index // STEP
labels[spike_window, Class - 1] = 1
# spike_window: which window each spike belongs to (one entry per spike)
# labels: one row per window, a 1 in the column of each class whose spike starts there
# two spikes in one window will be two 1s
windows = np.lib.stride_tricks.sliding_window_view(d, WIDTH)[::STEP]
# Convert to PyTorch tensors (X = network inputs, y = targets)
X = torch.tensor(windows, dtype=torch.float32)
y = torch.tensor(labels)


# Build the Network: 75 input > 50 > 25 > 5
model = nn.Sequential(
    nn.Linear(WIDTH, 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 5),
)
# Weight Function, Loss Function and Optimizer
pos_weight = (len(y) - y.sum(dim=0)) / y.sum(dim=0)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#training loop
for epoch in range(EPOCHS):
    order = torch.randperm(len(X))
    total_loss = 0
    for i in range(0, len(X), BATCH_SIZE):
        batch = order[i:i + BATCH_SIZE]
        logits = model(X[batch])
        loss = loss_fn(logits, y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)

# Prediction: any output above THRESHOLD = one spike, class = largest output
with torch.no_grad():
    probs = torch.sigmoid(model(X))
is_spike = probs.max(dim=1).values > THRESHOLD
pred_Index = starts[is_spike.numpy()]
pred_Class = probs[is_spike].argmax(dim=1).numpy() + 1

Check (temporary):
print(len(pred_Index), pred_Index[:10], pred_Class[:10])
