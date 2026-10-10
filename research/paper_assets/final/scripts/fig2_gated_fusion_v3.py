"""FIG 2 — Adaptive Gated Fusion v3 technical diagram.

Faithful to the actual implementation (backend/models/fusion_models.py
GatedFusionWide and research/final_experiment/phase4_search.py, the exact
trunk used by F4_generic_wce_bal_n0.0_s42):

  concat = [F_I ; F_T]                       (2048-d)
  gates  = Sigmoid(MLP(concat))              MLP: Dropout->FC 2048->256
                                             ->ReLU->Dropout->FC 256->2
  G_I, G_T = gates[:,0], gates[:,1]          (scalar per sample, in (0,1))
  gated   = [G_I * F_I ; G_T * F_T]
  logits  = FC(1024)+BN+ReLU+Drop -> FC(512)+BN+ReLU+Drop -> FC(3) -> Softmax

No invented components; no per-modality independent sigmoid projections
(the real gate network is a JOINT MLP over the concatenated features).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import apply_style, save_all

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

apply_style()

fig, ax = plt.subplots(figsize=(7.16, 3.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.32", linewidth=1.1)


def box(x, y, w, h, text, fc, ec="#444444", fs=8.6, bold=False, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            family="monospace" if mono else None)


def arrow(x1, y1, x2, y2, color="#333333", lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1))


# Column layout ------------------------------------------------------------
# Inputs
box(1, 40, 13, 8, "Image", "#D6EAF8", bold=True)
box(1, 8, 13, 8, "Text\n(tweet)", "#D6EAF8", bold=True)
arrow(14, 44, 18.5, 44)
arrow(14, 12, 18.5, 12)

# Encoders
box(18.5, 39, 17, 10, "EfficientNet-B0\n(frozen)", "#D4E6F1")
box(18.5, 7, 17, 10, "DistilBERT CLS\n(frozen)", "#D4E6F1")
arrow(35.5, 44, 40, 44)
arrow(35.5, 12, 40, 12)

# Features
box(40, 40.5, 15, 7, "$F_I\\in\\mathbb{R}^{1280}$", "#FCF3CF", fs=9)
box(40, 8.5, 15, 7, "$F_T\\in\\mathbb{R}^{768}$", "#FCF3CF", fs=9)

# Concat for gates
box(41, 26, 13, 6, "[ $F_I$ ; $F_T$ ]\n2048-d", "#E8DAEF", fs=8.4)
arrow(47.5, 40.5, 47.5, 32.4)
arrow(47.5, 15.5, 47.5, 26 - 0.4 + 0.0, color="#333333")
arrow(55.2, 29, 58.6, 29)

# Gate network
box(58.6, 22.5, 19.5, 13,
    "Gate Network\nDropout → FC 2048→256\n→ ReLU → Dropout →\nFC 256→2 → Sigmoid",
    "#FDEBD0", fs=7.8, bold=True)

# Gates out
arrow(78.1, 31, 82.3, 38.5)
arrow(78.1, 27, 82.3, 17.5)
box(82.3, 36.5, 8.2, 6.5, "$G_I$", "#FADBD8", fs=10)
box(82.3, 13, 8.2, 6.5, "$G_T$", "#FADBD8", fs=10)
arrow(90.5, 40, 93.2, 43.6)
arrow(90.5, 16, 93.2, 12.4)

# Gated features
box(93.2, 41.5, 6.6, 6.5, "$G_I{\\odot}F_I$", "#ABEBC6", fs=8.2)
box(93.2, 8.5, 6.6, 6.5, "$G_T{\\odot}F_T$", "#ABEBC6", fs=8.2)

# To classifier trunk
box(60, 3.2, 33, 3.6, "Concatenate gated features  [ $G_I{\\odot}F_I$ ; $G_T{\\odot}F_T$ ]  (2048-d)",
    "#D5F5E3", fs=8)
arrow(96.5, 8.5, 80, 6.9)
arrow(60, 5, 52, 5)

# Classifier trunk
box(24.5, 2.6, 27.5, 5.0,
    "FC 1024 · BatchNorm · ReLU · Dropout\nFC 512 · BatchNorm · ReLU · Dropout",
    "#D4E6F1", fs=7.8)
arrow(24.5, 5, 20.5, 5)
box(9, 2.6, 11.5, 5.0, "FC 3\nlogits", "#D4E6F1", fs=8.6)
arrow(9, 5.1, 5.5, 5.1)
box(0.5, 2.2, 5, 5.8, "Soft-\nmax", "#FADBD8", fs=8.4)

# Gate math annotation (top-right area)
ax.text(50, 53.5,
        r"Gates:  $[G_I,\;G_T]^{\top}"
        r"=\sigma\left(W_2\,\mathrm{ReLU}(W_1[F_I;F_T]+b_1)+b_2\right)\in(0,1)^2$"
        r"$\qquad$Gated features:  $\tilde{F}_I=G_I\odot F_I,\;\ \tilde{F}_T=G_T\odot F_T$",
        fontsize=8.6, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.45", fc="#FFFBF0", ec="#B7950B", lw=1.0))

# Output classes note under softmax
ax.text(3.0, 0.6,
        "Classes: Little/No · Mild · Severe Damage",
        fontsize=7.6, ha="left", va="center", style="italic", color="#555555")

paths = save_all(fig, "fig2_gated_fusion_v3")
print("fig2 done:", list(paths.values()))
