"""Generate the CrisisMind end-to-end architecture diagram (PNG).

Renders the updated dataset/input portion of the pipeline:
    Dataset (CrisisMMD v2.0)
      -> Data Quality Filtering            (audit-driven, objective rules)
      -> Quality-Controlled Dataset        (3,105 samples, duplicates removed)
      -> Stratified Split 70/15/15         (leak-free, near-duplicate containment)
      -> Train / Validation / Test         (distinguished; test excluded from training)
      -> Image + Text Processing           (EfficientNet-B0 1280-d + DistilBERT 768-d)
      -> Adaptive Gated Fusion v3
      -> Damage Severity Classification
      -> Emergency Decision Support
      -> Explainable AI / Dashboard

Run:  python generate_architecture_diagram.py
Output: Architecture@crisismind.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
NODE_W, NODE_H = 3.4, 0.72
X0 = 1.0
Y_TOP = 10.6
STEP = 1.28

COLORS = {
    "dataset":   "#D6E4F0",
    "filter":    "#FDEBD0",
    "clean":     "#D5F5E3",
    "split":     "#E8DAEF",
    "process":   "#FCF3CF",
    "model":     "#D4E6F1",
    "fusion":    "#EBDEF0",
    "classify":  "#FADBD8",
    "support":   "#D1F2EB",
    "xai":       "#FDEBD0",
    "edge":      "#5D6D7E",
    "label":     "#17202A",
}


def box(ax, x, y, w, h, text, color, title=None, title_color=None, dashed=False, fontsize=9):
    fc = COLORS[color]
    ec = COLORS["edge"]
    style = "--" if dashed else "-"
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.06,rounding_size=0.10",
                        linewidth=1.4, edgecolor=ec, facecolor=fc, linestyle=style)
    ax.add_patch(bb)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=COLORS["label"], wrap=True, zorder=5)
    if title:
        ax.text(x + w / 2, y + h + 0.07, title, ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=title_color or COLORS["label"])


def arrow(ax, x1, y1, x2, y2, label=None, dashed=False):
    style = "->"
    ls = "--" if dashed else "-"
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, mutation_scale=18,
                         linewidth=1.5, color=COLORS["edge"], linestyle=ls)
    ax.add_patch(ar)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.06, label, ha="center",
                va="bottom", fontsize=8, style="italic", color=COLORS["edge"])


def run():
    fig, ax = plt.subplots(figsize=(13.5, 13.0))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 13.4)
    ax.axis("off")

    W, H = NODE_W, NODE_H
    cx = X0 + W / 2  # center x of main spine

    y = Y_TOP
    box(ax, X0, y, W, H, "CrisisMMD v2.0\n18,082 annotation rows · 7 disasters",
        "dataset", title="1 · Dataset")
    y -= STEP

    box(ax, X0, y, W, H,
        "Data Quality Filtering\nobjective & reproducible rules\n(no prediction-based removal)",
        "filter", title="2 · Data Quality Filtering")
    y -= STEP

    box(ax, X0, y, W, H,
        "Quality-Controlled Dataset\n3,105 samples · duplicates removed\nconflicting/low-confidence labels excluded",
        "clean", title="3 · Quality-Controlled Dataset")
    y -= STEP

    box(ax, X0, y, W, H,
        "Stratified Split 70 / 15 / 15\nnear-duplicate containment · no train/test leakage",
        "split", title="4 · Stratified Split")
    y -= STEP

    # Three split lanes
    lane_w, lane_h = 2.2, 0.72
    lane_x = [0.4, 5.0, 9.6]
    lane_y = y
    lane_txt = {
        0: "TRAIN\n2,173 samples",
        1: "VALIDATION\n467 samples",
        2: "TEST\n465 samples",
    }
    for i, lx in enumerate(lane_x):
        box(ax, lx, lane_y, lane_w, lane_h, lane_txt[i], "split", fontsize=8.5)
    # connect split box to lanes
    for lx in lane_x:
        arrow(ax, cx, y + H + 0.02, lx + lane_w / 2, y, label=None)
    y -= STEP

    # processing (shared, but test flows around training)
    box(ax, X0, y, W, H,
        "Image + Text Processing\nResize 224 · tokenize ≤64 · frozen encoders\nEfficientNet-B0 (1280-d) + DistilBERT (768-d)",
        "process", title="5 · Image + Text Processing")
    y -= STEP

    box(ax, X0, y, W, H,
        "Adaptive Gated Fusion v3\nGatedFusionWide · learnable modality gates",
        "fusion", title="6 · Adaptive Gated Fusion v3")
    y -= STEP

    box(ax, X0, y, W, H,
        "Damage Severity Classification\nLittle / Mild / Severe\nweighted CE · gate regularization",
        "classify", title="7 · Damage Severity Classification")
    y -= STEP

    box(ax, X0, y, W, H,
        "Emergency Decision Support\nconfidence-aware severity triage",
        "support", title="8 · Emergency Decision Support")
    y -= STEP

    box(ax, X0, y, W, H,
        "Explainable AI · Dashboard\nGrad-CAM · attention · FastAPI + React",
        "xai", title="9 · Explainable AI / Dashboard")

    # ---- Arrows along spine ----
    yy = Y_TOP
    box_centers = [Y_TOP - i * STEP for i in range(9)]
    for i in range(len(box_centers) - 1):
        b1 = box_centers[i]
        b2 = box_centers[i + 1]
        arrow(ax, cx, b1 - 0.02, cx, b2 + H + 0.02)

    # ---- Train/Val path into processing ----
    lane_bottom = lane_y
    # TRAIN and VAL -> processing
    for lx in lane_x[:2]:
        arrow(ax, lx + lane_w / 2, lane_bottom - 0.02,
              cx, box_centers[4] + H + 0.02, label="train / val")
    # TEST -> evaluation only (bypasses training, dashed)
    arrow(ax, lane_x[2] + lane_w / 2, lane_bottom - 0.02,
          cx + W / 2 - 0.1, box_centers[5] + H + 0.02,
          label="test (evaluation only)", dashed=True)

    # annotate
    ax.text(cx, 12.35, "CrisisMind — Multimodal Disaster Damage Severity Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["label"])
    ax.text(cx, 0.28,
            "Gated Fusion v3 · EfficientNet-B0 + DistilBERT frozen encoders · "
            "leak-free data pipeline (2026-08-14)",
            ha="center", va="center", fontsize=9, color=COLORS["edge"])

    fig.tight_layout()
    fig.savefig("Architecture@crisismind.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Wrote Architecture@crisismind.png")


if __name__ == "__main__":
    run()
