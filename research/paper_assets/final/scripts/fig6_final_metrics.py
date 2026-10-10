"""FIG 6 — Final model performance summary (5 metrics, exact values).

Values read LIVE from FINAL_TEST_RESULTS.json and asserted against
FINAL_REPORT.md. No thresholds or targets are drawn.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from style import apply_style, save_all, load_final

EXPECTED = {"Accuracy": 69.50, "Macro Precision": 59.58, "Macro Recall": 54.80,
            "Macro F1": 56.50, "Weighted F1": 67.94}

final = load_final()
tm = final["test_metrics"]
vals = {
    "Accuracy": tm["accuracy"] * 100,
    "Macro\nPrecision": tm["precision_macro"] * 100,
    "Macro\nRecall": tm["recall_macro"] * 100,
    "Macro\nF1": tm["f1_macro"] * 100,
    "Weighted\nF1": tm["f1_weighted"] * 100,
}
for k, e in zip(["Accuracy", "Macro\nPrecision", "Macro\nRecall", "Macro\nF1", "Weighted\nF1"],
                EXPECTED.values()):
    assert abs(vals[k] - e) < 0.005, f"MISMATCH {k}: {vals[k]} vs {e}"
print("value assertions passed")

apply_style()
fig, ax = plt.subplots(figsize=(4.6, 2.9))
names = list(vals.keys())
v = [vals[n] for n in names]
colors = ["#1f77b4", "#8c564b", "#e377c2", "#ff7f0e", "#2ca02c"]
bars = ax.bar(names, v, color=colors, width=0.62)
for b, val in zip(bars, v):
    ax.text(b.get_x() + b.get_width() / 2, val + 1.0, f"{val:.2f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Score on held-out test set (%)")
ax.set_ylim(0, 82)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.tick_params(axis="x", labelsize=8.2)
ax.axhline(0, color="black", lw=0.8)
paths = save_all(fig, "fig6_final_metrics")
print("fig6 done")
