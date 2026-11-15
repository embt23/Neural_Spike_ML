"""Plot the overnight run's validation F at every check (every 25 epochs).

Run from the project root:
    python Big_model/plot_f.py
Saves Artifacts/Big_Model_F.png.
"""
import json

import numpy as np
import matplotlib.pyplot as plt

rows = [json.loads(line) for line in open("Big_model/data/log.jsonl")]
epochs = np.array([r["epoch"] for r in rows])
f = np.array([r["f"] for r in rows])
best = int(np.argmax(f))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, f, color="#2a78d6", linewidth=1.5)
ax.plot(epochs[best], f[best], "o", color="#2a78d6", markersize=8)
ax.annotate(f"best: F {f[best]:.3f} at epoch {epochs[best]}",
            (epochs[best], f[best]), xytext=(8, 6), textcoords="offset points")

ax.set_ylim(top=f.max() + 0.01)   # room for the label
ax.set_xlabel("Epoch")
ax.set_ylabel("Best F (validation, level 6)")
ax.set_title("Validation F over the overnight run (50-100-50-25-1, POS_WEIGHT 5, 9 h)")
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.savefig("Artifacts/Big_Model_F.png", dpi=120)
plt.show()
