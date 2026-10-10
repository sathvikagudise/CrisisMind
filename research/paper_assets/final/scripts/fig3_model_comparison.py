"""FIG 3 — Final multimodal ablation / model comparison.

Values are read LIVE from:
  - backend/outputs/checkpoints/final_exp/ablation_results.json  (img/txt/concat/v1/v2)
  - backend/outputs/checkpoints/final_exp/seed_runs.json         (final v3, seed-42 frozen eval)

Assertions pin every plotted number to the values recorded in
FINAL_REPORT.md; the script fails loudly on any mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (apply_style, save_all, load_ablation, load_seeds,
                   C_ACC, C_F1, C_V3)

import numpy as np
import matplotlib.pyplot as plt

EXPECTED = {  # model -> (accuracy %, macro-F1 %) as recorded in FINAL_REPORT.md
    "Image-Only":              (64.68, 53.35),
    "Text-Only":               (53.67, 46.25),
    "Concatenation":           (64.68, 56.18),
    "Gated Fusion v1":         (58.72, 54.05),
    "Gated Fusion v2":         (58.72, 54.05),
    "Adaptive Gated Fusion v3\n(proposed)": (69.50, 56.50),
}

abl = load_ablation()
seeds = load_seeds()
vals = {
    "Image-Only":              (abl["img"]["test"]["accuracy"] * 100,
                                abl["img"]["test"]["f1_macro"] * 100),
    "Text-Only":               (abl["txt"]["test"]["accuracy"] * 100,
                                abl["txt"]["test"]["f1_macro"] * 100),
    "Concatenation":           (abl["concat"]["test"]["accuracy"] * 100,
                                abl["concat"]["test"]["f1_macro"] * 100),
    "Gated Fusion v1":         (abl["v1"]["test"]["accuracy"] * 100,
                                abl["v1"]["test"]["f1_macro"] * 100),
    "Gated Fusion v2":         (abl["v2"]["test"]["accuracy"] * 100,
                                abl["v2"]["test"]["f1_macro"] * 100),
    "Adaptive Gated Fusion v3\n(proposed)": (
        seeds["F4_generic_wce_bal_n0.0"]["test_at_seed42"]["accuracy"] * 100,
        seeds["F4_generic_wce_bal_n0.0"]["test_at_seed42"]["f1_macro"] * 100),
}
for k, (a, f) in vals.items():
    ea, ef = EXPECTED[k]
    assert abs(a - ea) < 0.005 and abs(f - ef) < 0.005, f"MISMATCH {k}: got {(a, f)}"
print("value assertions passed")

apply_style()
fig, ax = plt.subplots(figsize=(7.16, 3.0))
names = list(vals.keys())
acc = [vals[n][0] for n in names]
f1 = [vals[n][1] for n in names]
x = np.arange(len(names))
w = 0.38
bars_a = ax.bar(x - w / 2, acc, w, label="Accuracy", color=C_ACC,
                edgecolor=C_V3 if False else "black", linewidth=0.0)
bars_f = ax.bar(x + w / 2, f1, w, label="Macro F1", color=C_F1)
# Emphasize proposed model only via edge highlight + hatch-free bold tick
bars_a[-1].set_edgecolor("black"); bars_a[-1].set_linewidth(1.4)
bars_f[-1].set_edgecolor("black"); bars_f[-1].set_linewidth(1.4)
for b, v in list(zip(bars_a, acc)) + list(zip(bars_f, f1)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f}",
            ha="center", va="bottom", fontsize=7.6)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8.2)
ax.get_xticklabels()[-1].set_fontweight("bold")
ax.set_ylabel("Test set score (%)")
ax.set_ylim(0, 80)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ax.axhline(0, color="black", lw=0.8)
ax.legend(loc="upper left", frameon=False, ncol=2)
paths = save_all(fig, "fig3_model_comparison")
print("fig3 done")
