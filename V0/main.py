import time
import numpy as np
import torch
from torch import nn
from grading import grade
import json
import os


# ---------------- Settings ----------------
FS = 25000              # sampling frequency (Hz)
STEP = 10               # how many samples the window jumps each time
WIDTH = 75              # how many samples each window looks at (= input layer size)
HIDDEN = [50, 25]       # hidden layer sizes
POS_WEIGHT = 300        # a missed spike counts this many times more than a false alarm
LEARNING_RATE = 0.001   # how big each update nudge is
EPOCHS = 20             # how many full passes through the training data
BATCH_SIZE = 256        # how many windows per update
SEED = 0                # fixes the random start so runs can be repeated exactly
THRESHOLD = 0.5         # output above this = spike
RESULTS_DIR = "Results" # folder for saved models and the run log


# ---------------- Functions ----------------
def make_windows(d):
    """Cut d into WIDTH-sample windows every STEP samples.
    Returns each window's left edge and the windows as a tensor."""
    starts = np.arange(0, len(d) - WIDTH + 1, STEP)
    windows = np.lib.stride_tricks.sliding_window_view(d, WIDTH)[::STEP]
    return starts, torch.tensor(windows, dtype=torch.float32)


def make_labels(n_windows, Index, Class):
    """One row per window: a 1 in the class column if that class's spike starts
    in the window's first STEP samples, otherwise all 0."""
    labels = np.zeros((n_windows, 5), dtype=np.float32)
    spike_window = Index // STEP
    labels[spike_window, Class - 1] = 1
    return torch.tensor(labels)


def build_model():
    """WIDTH inputs -> HIDDEN layers (ReLU) -> 5 outputs (logits)."""
    layers = []
    n_in = WIDTH
    for n_out in HIDDEN:
        layers.append(nn.Linear(n_in, n_out))
        layers.append(nn.ReLU())
        n_in = n_out
    layers.append(nn.Linear(n_in, 5))
    return nn.Sequential(*layers)


def train(model, X, y):
    """Forward -> loss -> backward -> update, for EPOCHS passes over shuffled batches."""
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
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
        losses.append(total_loss / len(X))
        print(f"epoch {epoch:3d}   loss {losses[-1]:.4f}")
    return losses

def predict(model, d):
    """Any output above THRESHOLD = one spike at the window's left edge,
    class = largest output. Returns 0-based Index and Class 1-5."""
    starts, X = make_windows(d)
    with torch.no_grad():
        probs = torch.sigmoid(model(X))
    is_spike = probs.max(dim=1).values > THRESHOLD
    pred_Index = starts[is_spike.numpy()]
    pred_Class = probs[is_spike].argmax(dim=1).numpy() + 1
    return pred_Index, pred_Class


# ---------------- Main ----------------
if __name__ == "__main__":
    torch.manual_seed(SEED)
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Load the prepared D1 data (Index already 0-based and sorted)
    train_data = np.load("Data/D1_train.npz")
    test_data = np.load("Data/D1_test.npz")

    # Training windows and targets
    starts, X = make_windows(train_data["d"])
    y = make_labels(len(starts), train_data["Index"], train_data["Class"])

    # Build and train, timing the training
    model = build_model()
    t0 = time.perf_counter()
    losses = train(model, X, y)
    train_time = time.perf_counter() - t0

    # Score on training data and test data
    pred_Index, pred_Class = predict(model, train_data["d"])
    train_score = grade(train_data["Index"], train_data["Class"], pred_Index, pred_Class)

    t0 = time.perf_counter()
    pred_Index, pred_Class = predict(model, test_data["d"])
    test_predict_time = time.perf_counter() - t0
    test_score = grade(test_data["Index"], test_data["Class"], pred_Index, pred_Class)

    print("train:", train_score)
    print("test: ", test_score)


    # Save the trained model and a record of this run
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{RESULTS_DIR}/{run_name}.pt")
    record = {
        "run": run_name,
        "settings": {"STEP": STEP, "WIDTH": WIDTH, "HIDDEN": HIDDEN, "LEARNING_RATE": LEARNING_RATE,
                     "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "POS_WEIGHT": POS_WEIGHT,
                     "THRESHOLD": THRESHOLD, "SEED": SEED},
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "train_time_s": train_time,
        "test_predict_time_s": test_predict_time,
        "losses": losses,
        "train_score": train_score,
        "test_score": test_score,
    }
    with open(f"{RESULTS_DIR}/results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Saved {RESULTS_DIR}/{run_name}.pt and added a line to {RESULTS_DIR}/results.jsonl")
