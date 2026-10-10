"""Utility functions for the explainability module."""

import os
import uuid
from pathlib import Path
from backend.config.shared import OUTPUTS_DIR

EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"


def make_explanations_dir():
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return EXPLANATIONS_DIR


def make_explanation_paths(analysis_id=None):
    """Return (original_path, heatmap_path, overlay_path) for an analysis."""
    make_explanations_dir()
    aid = analysis_id or str(uuid.uuid4())
    original = EXPLANATIONS_DIR / f"{aid}_original.png"
    heatmap = EXPLANATIONS_DIR / f"{aid}_heatmap.png"
    overlay = EXPLANATIONS_DIR / f"{aid}_overlay.png"
    return aid, original, heatmap, overlay


def save_explanation_images(original_pil, heatmap_pil, overlay_pil,
                            analysis_id=None):
    """Save the three Grad-CAM images and return their relative URLs."""
    aid, orig_path, heat_path, over_path = make_explanation_paths(analysis_id)
    original_pil.save(orig_path, "PNG")
    heatmap_pil.save(heat_path, "PNG")
    overlay_pil.save(over_path, "PNG")
    urls = {
        "original": f"/outputs/explanations/{orig_path.name}",
        "heatmap": f"/outputs/explanations/{heat_path.name}",
        "overlay": f"/outputs/explanations/{over_path.name}",
    }
    return aid, urls
