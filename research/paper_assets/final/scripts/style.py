"""Shared IEEE publication style for all final paper figures.

Every quantitative figure reads its numbers LIVE from the authoritative
artifacts under backend/outputs/checkpoints/final_exp/ so that figures
provably match FINAL_TEST_RESULTS.json / FINAL_REPORT.md.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(HERE)
ROOT = os.path.normpath(os.path.join(FINAL_DIR, "..", "..", ".."))
EXP_DIR = os.path.join(ROOT, "backend", "outputs", "checkpoints", "final_exp")

FINAL_JSON = os.path.join(EXP_DIR, "FINAL_TEST_RESULTS.json")
ABLATION_JSON = os.path.join(EXP_DIR, "ablation_results.json")
SEEDS_JSON = os.path.join(EXP_DIR, "seed_runs.json")


def load_final():
    with open(FINAL_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_ablation():
    with open(ABLATION_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_seeds():
    with open(SEEDS_JSON, encoding="utf-8") as f:
        return json.load(f)


def apply_style():
    plt.rcParams.update({
        # IEEE-friendly: readable at 3.5 in single-column print
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


# Consistent palette (colorblind-safe accents)
C_ACC = "#1f77b4"   # accuracy / primary
C_F1 = "#ff7f0e"    # macro-F1 / secondary
C_V3 = "#2ca02c"    # proposed model emphasis
CLASS_NAMES = ["Little/No Damage", "Mild Damage", "Severe Damage"]


def save_all(fig, name):
    """Save PNG (+PDF+SVG) into research/paper_assets/final/."""
    paths = {}
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(FINAL_DIR, f"{name}.{ext}")
        fig.savefig(p, facecolor="white")
        paths[ext] = p
    plt.close(fig)
    for e, p in paths.items():
        print(f"  wrote {os.path.relpath(p, ROOT)} ({os.path.getsize(p)} bytes)")
    return paths
