"""Explainable AI module — Grad-CAM for image-based predictions."""

from backend.explainability.gradcam import GradCAM, TARGET_LAYER
from backend.explainability.service import generate_explanation
from backend.explainability.image_overlay import overlay_heatmap
from backend.explainability.utils import (
    save_explanation_images,
    make_explanations_dir,
)

__all__ = [
    "GradCAM",
    "TARGET_LAYER",
    "generate_explanation",
    "overlay_heatmap",
    "save_explanation_images",
    "make_explanations_dir",
]
