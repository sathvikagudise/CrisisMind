"""API-specific configuration.

Re-exports shared constants from backend.config.shared and adds
settings exclusive to the prediction API.
"""

from backend.config.shared import (
    DEVICE,
    CHECKPOINTS_DIR,
    IMG_CKPT,
    TEXT_CKPT,
    GATED_CKPT,
    IMG_SIZE,
    TEXT_MAX_LEN,
    TEXT_MODEL_NAME,
    FUSION_DROPOUT,
    NUM_CLASSES,
)

MAX_IMAGE_SIZE_MB = 10
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

SEVERITY_MAP = {0: "Little or No Damage", 1: "Mild Damage", 2: "Severe Damage"}
