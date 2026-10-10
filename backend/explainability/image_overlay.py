"""Heatmap-to-image overlay utilities."""

import numpy as np
from PIL import Image


def overlay_heatmap(original_pil, heatmap_np, alpha=0.5, colormap="jet"):
    """Overlay a heatmap (H, W) on a PIL image and return a PIL image.

    Args:
        original_pil: PIL Image in RGB.
        heatmap_np: 2-D numpy array in [0, 1] (H, W).
        alpha: Blending factor for the heatmap.
        colormap: Matplotlib colormap name.

    Returns:
        PIL Image (RGB) with the heatmap overlaid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(heatmap_np)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    heatmap_pil = Image.fromarray(heatmap_rgb).resize(
        original_pil.size, Image.BILINEAR
    )
    blended = Image.blend(original_pil.convert("RGB"), heatmap_pil, alpha)
    return blended


def heatmap_to_pil(heatmap_np, colormap="jet"):
    """Convert a heatmap array to a PIL image (resized to 224x224)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(heatmap_np)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(heatmap_rgb).resize((224, 224), Image.BILINEAR)
