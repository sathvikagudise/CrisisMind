"""Load all models required by the Analysis Engine."""

import torch
import logging

from backend.models.fusion_models import (
    ImageFeatureExtractor,
    TextFeatureExtractor,
    GatedFusionWide,
)
from backend.models.efficientnet import build_model as build_image_classifier
from backend.models.text_model import build_text_model
from backend.api.config import (
    DEVICE,
    IMG_CKPT,
    TEXT_CKPT,
    GATED_CKPT,
    FUSION_DROPOUT,
    NUM_CLASSES,
)
from backend.config.shared import FUSION_DIM_IMG, FUSION_DIM_TXT, FUSION_ENC_MODE

logger = logging.getLogger("crisismind")


def load_all_models() -> dict:
    logger.info("AnalysisEngine: loading models on %s", DEVICE.upper())

    img_enc_path = None if FUSION_ENC_MODE == "generic" else IMG_CKPT
    txt_enc_path = None if FUSION_ENC_MODE == "generic" else TEXT_CKPT
    logger.info("  [1/5] Loading ImageFeatureExtractor (EfficientNet-B0, mode=%s)...", FUSION_ENC_MODE)
    img_encoder = ImageFeatureExtractor(img_enc_path, freeze=True).to(DEVICE)
    img_encoder.eval()

    logger.info("  [2/5] Loading TextFeatureExtractor (DistilBERT, mode=%s)...", FUSION_ENC_MODE)
    txt_encoder = TextFeatureExtractor(txt_enc_path, freeze=True).to(DEVICE)
    txt_encoder.eval()

    logger.info("  [3/5] Loading GatedFusionWide classifier (v3)...")
    gated = GatedFusionWide(
        img_dim=FUSION_DIM_IMG,
        txt_dim=FUSION_DIM_TXT,
        num_classes=NUM_CLASSES,
        dropout=FUSION_DROPOUT,
    )
    state = torch.load(GATED_CKPT, map_location=DEVICE)
    gated.load_state_dict(state)
    gated.to(DEVICE)
    gated.eval()

    logger.info("  [4/5] Loading EfficientNet classifier (image-only)...")
    image_clf = build_image_classifier(num_classes=NUM_CLASSES, freeze_backbone=True)
    img_state = torch.load(IMG_CKPT, map_location=DEVICE)
    image_clf.load_state_dict(img_state, strict=False)
    image_clf.to(DEVICE)
    image_clf.eval()

    logger.info("  [5/5] Loading DistilBERT classifier (text-only)...")
    text_clf = build_text_model(num_classes=NUM_CLASSES, freeze_bert=True)
    txt_state = torch.load(TEXT_CKPT, map_location=DEVICE)
    text_clf.load_state_dict(txt_state, strict=False)
    text_clf.to(DEVICE)
    text_clf.eval()

    logger.info("AnalysisEngine: all 5 models loaded successfully.")
    return {
        "img_encoder": img_encoder,
        "txt_encoder": txt_encoder,
        "gated_model": gated,
        "image_classifier": image_clf,
        "text_classifier": text_clf,
    }
