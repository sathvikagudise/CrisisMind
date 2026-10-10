# Explainability Module — Grad-CAM

## Overview

This module implements **Grad-CAM (Gradient-weighted Class Activation Mapping)** for the CrisisMind EfficientNet-B0 image classifier. It generates heatmaps that highlight which regions of an input image most influenced the model's severity prediction (Little or No Damage / Mild Damage / Severe Damage).

## Architecture

```
explainability/
├── __init__.py          # Package exports
├── gradcam.py           # Core Grad-CAM implementation
├── service.py           # Service orchestration (image-only & multimodal)
├── utils.py             # Path/IO utilities
├── image_overlay.py     # Heatmap-to-image overlay
└── README.md
```

## Components

### `GradCAM` (gradcam.py)

Hooks the last convolutional block (`features.8`) of EfficientNet-B0. For each input image:
1. Forward pass to capture activation maps (7×7 spatial).
2. Backward pass from the predicted class score to capture gradients.
3. Weight activations by averaged gradients → ReLU → normalize → upsample to 224×224.

### `generate_explanation` (service.py)

Two pathways depending on input:

- **Image-only**: Forward through `image_classifier` → Grad-CAM on `features.8`.
- **Multimodal**: Forward through `img_encoder` → `gated_model` → Grad-CAM on `img_encoder.backbone.features.8` (gradient flows through the gated fusion head).

### Output

Three images are saved per explanation:

| Image | Description |
|-------|-------------|
| `original` | The input image (as-received) |
| `heatmap` | Raw Grad-CAM heatmap (jet colormap, 224×224) |
| `overlay` | 50% blended heatmap over original |

Saved to `backend/outputs/explanations/{analysis_id}_{type}.png`.

## API

```
POST /explainability/gradcam
```

Multipart form:
- `image` (required) — JPEG/PNG/WebP/BMP, max 10 MB
- `tweet_text` (optional) — if provided, uses multimodal pipeline
- `analysis_id` (optional) — link to prior analysis

Returns:
```json
{
  "success": true,
  "analysis_id": "...",
  "analysis_mode": "image_only|multimodal",
  "prediction": {
    "severity": "Severe Damage",
    "confidence": 91.5
  },
  "images": {
    "original": "/outputs/explanations/abc_original.png",
    "heatmap": "/outputs/explanations/abc_heatmap.png",
    "overlay": "/outputs/explanations/abc_overlay.png"
  },
  "explanation": {
    "technique": "Grad-CAM",
    "target_layer": "EfficientNet-B0.features.8",
    "generated_at": "2025-07-09T19:30:00Z"
  }
}
```

## Target Layer

`EfficientNet-B0.features.8` — the final `Conv2dNormActivation` block (320→1280 channels, 1×1 conv) that produces 7×7 feature maps before global average pooling.

## Integration

- **Frontend**: "AI Explanation" button on the Manual Analysis page → 3 image cards (Original, Heatmap, Overlay).
- **Reports**: If an overlay exists (`outputs/explanations/{id}_overlay.png`), the PDF generator embeds it in the report.
- **Text-only mode**: Button is disabled with a message.

## Requirements

- Uses the existing `image_classifier` (image-only) and `img_encoder` + `gated_model` (multimodal) from the Analysis Engine.
- No model retraining or weight modification.
- Gradient computation requires a `torch.enable_grad()` context during the forward pass.
