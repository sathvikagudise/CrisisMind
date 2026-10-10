"""Explainability service — generates Grad-CAM for image & multimodal predictions."""

import logging
from datetime import datetime
from contextlib import nullcontext

import torch
import numpy as np
from PIL import Image

from backend.explainability.gradcam import GradCAM, TARGET_LAYER
from backend.explainability.image_overlay import overlay_heatmap, heatmap_to_pil
from backend.explainability.utils import save_explanation_images
from backend.api.preprocessing import preprocess_image, tokenize_text
from backend.api.config import DEVICE, SEVERITY_MAP

logger = logging.getLogger("crisismind.explainability")


def _enable_grad(model):
    """Temporarily enable gradient flow on frozen models for Grad-CAM."""
    for p in model.parameters():
        p.requires_grad = True


def _prediction_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred_idx].item() * 100)
    severity = SEVERITY_MAP[pred_idx]
    return severity, round(confidence, 2)


def generate_explanation(engine, image: Image.Image,
                         tweet_text: str = None,
                         analysis_id: str = None) -> dict:
    """Generate a Grad-CAM explanation for an image.

    Args:
        engine: AnalysisEngine instance (provides models).
        image: PIL Image to explain.
        tweet_text: Optional text for multimodal Grad-CAM.
        analysis_id: Optional ID for saving.

    Returns:
        Dict matching the API response schema.
    """
    img_tensor = preprocess_image(image).to(DEVICE)
    original_size = image.size

    if tweet_text:
        logger.info("Generating multimodal Grad-CAM")
        img_encoder = engine.models["img_encoder"]
        txt_encoder = engine.models["txt_encoder"]
        gated_model = engine.models["gated_model"]

        _enable_grad(img_encoder)
        _enable_grad(gated_model)

        gradcam = GradCAM(img_encoder.backbone, target_layer=TARGET_LAYER)

        input_ids, attention_mask = tokenize_text(tweet_text)
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)

        img_encoder.zero_grad()
        gradcam.activations = None
        gradcam.gradients = None

        with torch.enable_grad():
            img_feat = img_encoder(img_tensor)
            txt_feat = txt_encoder(input_ids, attention_mask)
            logits, gates = gated_model(img_feat, txt_feat)

        class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        activations = gradcam.activations[0]
        gradients = gradcam.gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        heatmap = torch.sum(weights * activations, dim=0)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap_np = heatmap.cpu().numpy()

        heatmap_resized = np.array(
            Image.fromarray(heatmap_np).resize(original_size, Image.BILINEAR)
        )
        overlay_pil = overlay_heatmap(image, heatmap_resized, alpha=0.5)
        heatmap_pil = heatmap_to_pil(heatmap_np)

        severity, confidence = _prediction_from_logits(logits)
        analysis_mode = "multimodal"
    else:
        logger.info("Generating image-only Grad-CAM")
        classifier = engine.models["image_classifier"]
        _enable_grad(classifier)

        gradcam = GradCAM(classifier, target_layer=TARGET_LAYER)

        classifier.zero_grad()
        gradcam.activations = None
        gradcam.gradients = None

        with torch.enable_grad():
            logits = classifier(img_tensor)

        class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        activations = gradcam.activations[0]
        gradients = gradcam.gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        heatmap = torch.sum(weights * activations, dim=0)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap_np = heatmap.cpu().numpy()

        heatmap_resized = np.array(
            Image.fromarray(heatmap_np).resize(original_size, Image.BILINEAR)
        )
        overlay_pil = overlay_heatmap(image, heatmap_resized, alpha=0.5)
        heatmap_pil = heatmap_to_pil(heatmap_np)

        severity, confidence = _prediction_from_logits(logits)
        analysis_mode = "image_only"

    aid, urls = save_explanation_images(image, heatmap_pil, overlay_pil,
                                         analysis_id=analysis_id)

    return {
        "success": True,
        "analysis_id": aid,
        "analysis_mode": analysis_mode,
        "prediction": {
            "severity": severity,
            "confidence": confidence,
        },
        "images": urls,
        "explanation": {
            "technique": "Grad-CAM",
            "target_layer": f"EfficientNet-B0.{TARGET_LAYER}",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    }
