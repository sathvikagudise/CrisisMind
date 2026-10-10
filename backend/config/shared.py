"""Shared configuration constants shared between API and training modules.

All paths are resolved relative to this file's location, making imports
portable regardless of PYTHONPATH or working directory.
"""

import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Directory layout (resolved from backend/config/ → backend/)
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BACKEND_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# ---------------------------------------------------------------------------
# Model architecture constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 3
IMG_SIZE = 224
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_MAX_LEN = 64
FUSION_DROPOUT = 0.3
FUSION_DIM_IMG = 1280
FUSION_DIM_TXT = 768

# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
IMG_CKPT = CHECKPOINTS_DIR / "best_model.pth"
TEXT_CKPT = CHECKPOINTS_DIR / "best_text_model.pth"
# Deployed fusion head: authoritative final leakage-free checkpoint (F4_generic)
GATED_CKPT = CHECKPOINTS_DIR / "final_exp" / "F4_generic_wce_bal_n0.0_s42" / "best.pth"
CONCAT_CKPT = CHECKPOINTS_DIR / "best_concat_fusion.pth"

# Fusion feature-encoder mode (deployment parity with the reported F4_generic eval):
#   "generic" -> frozen ImageNet-vanilla EfficientNet-B0 + vanilla DistilBERT
#                (exactly what produced the reported leakage-free results)
#   "task"    -> legacy task-tuned best_model.pth / best_text_model.pth encoders
FUSION_ENC_MODE = "generic"
