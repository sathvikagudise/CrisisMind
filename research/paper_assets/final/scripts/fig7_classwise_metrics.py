"""FIG 7 — Class-wise Precision / Recall / F1 (exact final values).

Values read LIVE from FINAL_TEST_RESULTS.json per_class block and asserted
against FINAL_REPORT.md.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import apply_style, save_all, load_final, CLASS_NAMES

EXPECTED = {  # class -> (P, R, F1) %
    "Little/No Damage": (55.81, 39.34, 46.15),
    "Mild Damage":      (45.12, 37.37, 40.88),
    "Severe Damage":    (77.81, 87.68, 82.45),
}

final = load_final()
pc = final["per_class"]
key_map = {"Little/No Damage": "little_or_no_damage",
           "Mild Damage": "mild_damage",
           "Severe Damage": "severe_damage"}
vals = {c: (pc[key_map[c]]["precision"] * 100,
            pc[key_map[c]]["recall"] * 100,
            pc[key_map[c]]["f1"] * 100) for c in CLASS_NAMES}
for c, e in EXPECTED.items():
    got = tuple(round(x, 2) for x in vals[c])
    assert all(abs(g - x) < 0.005 for g, x in zip(got, e)), f"MISMATCH {c}: {got}"
print("value assertions passed")

apply_style()
fig, ax = plt.subplots(figsize=(4.6, 2.9))
x = np.arange(3)
w = 0.26
series = [("Precision", [vals[c][0] for c in CLASS_NAMES], "#1f77b4"),
          ("Recall",   [vals[c][1] for c in CLASS_NAMES], "#ff7f0e"),
          ("F1-score", [vals[c][2] for c in CLASS_NAMES], "#2ca02c")]
for k, (label, arr, color) in enumerate(series):
    bars = ax.bar(x + (k - 1) * w, arr, w, label=label, color=color)
    for b, v in zip(bars, arr):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.4)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=8.4)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=7.8)
paths = save_all(fig, "fig7_classwise_metrics")
print("fig7 done")
