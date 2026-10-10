"""FIG 4 — Final confusion matrix (leakage-free frozen test evaluation).

Values read LIVE from FINAL_TEST_RESULTS.json; asserted against the exact
matrix recorded in FINAL_REPORT.md.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import apply_style, save_all, load_final, CLASS_NAMES

EXPECTED = np.array([[24, 19, 18], [11, 37, 51], [8, 26, 242]])

final = load_final()
cm = np.array(final["confusion_matrix"])
assert (cm == EXPECTED).all(), f"CM MISMATCH: {cm}"
support = cm.sum(axis=1)
assert list(support) == [61, 99, 276] and cm.sum() == 436
print("value assertions passed")

apply_style()
fig, ax = plt.subplots(figsize=(4.6, 3.7))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Number of test samples", fontsize=8.5)
cbar.ax.tick_params(labelsize=8)

# Cell annotations: count + row-normalised recall %
for i in range(3):
    for j in range(3):
        cnt = cm[i, j]
        rec = 100.0 * cnt / support[i]
        color = "white" if cnt > cm.max() * 0.55 else "#1a1a1a"
        ax.text(j, i, f"{cnt}\n({rec:.1f}%)", ha="center", va="center",
                fontsize=9.5, color=color, fontweight="bold")

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(CLASS_NAMES, support)], fontsize=8.2)
ax.set_yticklabels([f"{c}\n(n={s})" for c, s in zip(CLASS_NAMES, support)], fontsize=8.2)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion matrix of the final leakage-free\nCrisisMind test evaluation (n = 436)",
             fontsize=9.5)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
paths = save_all(fig, "fig4_confusion_matrix")
print("fig4 done")
