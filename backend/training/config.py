"""Training-specific configuration.

Re-exports shared constants from backend.config.shared and adds
settings exclusive to model training.  Directory creation happens at
import time so that output directories always exist before training.
"""

from pathlib import Path

from backend.config.shared import (
    BACKEND_DIR,
    DEVICE,
    OUTPUTS_DIR,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    IMG_CKPT,
    TEXT_CKPT,
    IMG_SIZE,
    TEXT_MODEL_NAME,
    TEXT_MAX_LEN,
    FUSION_DROPOUT,
    NUM_CLASSES,
)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_DIR = Path("dataset_analysis/CrisisMMD_v2.0")
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
DATA_IMAGE_DIR = DATASET_DIR / "data_image"

# ---------------------------------------------------------------------------
# Label mapping (shared by evaluation scripts)
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "little_or_no_damage": 0,
    "mild_damage": 1,
    "severe_damage": 2,
}
LABEL_NAMES = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}

# ---------------------------------------------------------------------------
# Image training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 7
RANDOM_SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ---------------------------------------------------------------------------
# Text training hyper-parameters
# ---------------------------------------------------------------------------
TEXT_BATCH_SIZE = 16
TEXT_EPOCHS = 30
TEXT_LEARNING_RATE = 2e-5
TEXT_PATIENCE = 5

# ---------------------------------------------------------------------------
# Fusion training hyper-parameters
# ---------------------------------------------------------------------------
FUSION_BATCH_SIZE = 4
FUSION_EPOCHS = 15
FUSION_LR = 1e-3
FUSION_PATIENCE = 3

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
for d in [CHECKPOINTS_DIR, METRICS_DIR, REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
