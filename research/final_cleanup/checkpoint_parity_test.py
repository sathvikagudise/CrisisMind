"""Phase 5: deterministic parity test.

Compares, for the same test samples, two pipelines:
  A) EXACT evaluation path:  F4 head (best.pth) x cached generic features
     (features_generic.pt -> img_feats/txt_feats)  == the reported 69.50% eval.
  B) Deployed API path:      backend.api.preprocessing (image+text) ->
     generic ImageNet-vanilla B0 encoder -> vanilla DistilBERT encoder ->
     deployed GatedFusionWide (F4 head) -> softmax -> argmax.

Claimed result: A and B agree on predicted class, full class probabilities,
confidence, and fused logits within floating-point tolerance, i.e. the deployed
app reproduces the reported evaluation for the sampled inputs.
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from backend.config.shared import GATED_CKPT, FUSION_DIM_IMG, FUSION_DIM_TXT, NUM_CLASSES
from backend.models.fusion_models import GatedFusionWide
from backend.api.preprocessing import preprocess_image, tokenize_text
from backend.api.config import SEVERITY_MAP

DATASET = os.path.join(ROOT, "dataset_analysis", "CrisisMMD_v2.0")
SPLIT_CSV = os.path.join(ROOT, "research", "optimization", "cleaned_dataset.csv")
FEATURES = os.path.join(ROOT, "backend", "outputs", "checkpoints", "final_exp", "features_generic.pt")

DEVICE = "cpu"

# --- load F4 head and cached generic features (exact eval inputs) ---
head = GatedFusionWide(img_dim=FUSION_DIM_IMG, txt_dim=FUSION_DIM_TXT,
                       num_classes=NUM_CLASSES, dropout=0.2)
head.load_state_dict(torch.load(GATED_CKPT, map_location=DEVICE), strict=True)
head.to(DEVICE).eval()

cache = torch.load(FEATURES, map_location=DEVICE, weights_only=False)
img_feats, txt_feats, labels = cache["img_feats"], cache["txt_feats"], cache["labels"]
idx_test = cache["idx_test"]
paths = cache["paths"]

df = pd.read_csv(SPLIT_CSV)
# pick deterministic samples: first 12 test rows with existing images
samples = []
for i in idx_test:
    p = os.path.join(DATASET, paths[i])
    if os.path.exists(p):
        samples.append((int(i), paths[i], str(df.iloc[i]["tweet_text"]), int(labels[i])))
    if len(samples) == 12:
        break
print(f"parity samples: {len(samples)}")

# --- deployed encoders (generic, built identically to the API loader) ---
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import DistilBertModel
img_enc = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
img_enc.classifier = torch.nn.Identity()
img_enc.eval()
txt_enc = DistilBertModel.from_pretrained("distilbert-base-uncased").eval()

results = []
max_dev_class = 0
max_dev_prob = 0.0
max_dev_conf = 0.0

for gi, path, text, y_true in samples:
    # --- Path A: exact eval inputs (cached generic features) through F4 head ---
    with torch.no_grad():
        logits_a, gates_a = head(img_feats[gi:gi+1], txt_feats[gi:gi+1])
    prob_a = F.softmax(logits_a, dim=1)[0]
    cls_a = int(logits_a.argmax(1).item())
    conf_a = float(prob_a[cls_a].item() * 100)

    # --- Path B: deployed API preprocessing + generic encoders + F4 head ---
    im = Image.open(os.path.join(DATASET, path)).convert("RGB")
    with torch.no_grad():
        imt = preprocess_image(im)
        ids, mask = tokenize_text(text)
        img_feat = img_enc(imt)
        txt_feat = txt_enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        logits_b, gates_b = head(img_feat, txt_feat)
    prob_b = F.softmax(logits_b, dim=1)[0]
    cls_b = int(logits_b.argmax(1).item())
    conf_b = float(prob_b[cls_b].item() * 100)

    dev_class = 0 if cls_a == cls_b else 1
    dev_prob = float((prob_a - prob_b).abs().max().item())
    dev_conf = abs(conf_a - conf_b)
    dev_logit = float((logits_a - logits_b).abs().max().item())
    dev_gate = float((gates_a - gates_b).abs().max().item())
    dev_feat_img = float((img_feats[gi] - img_feat[0]).abs().max().item())
    dev_feat_txt = float((txt_feats[gi] - txt_feat[0]).abs().max().item())

    max_dev_class = max(max_dev_class, dev_class)
    max_dev_prob = max(max_dev_prob, dev_prob)
    max_dev_conf = max(max_dev_conf, dev_conf)

    results.append({
        "sample": gi, "path": path, "truth": int(y_true),
        "cls_A": cls_a, "cls_B": cls_b, "dev_class": dev_class,
        "prob_A": [round(x, 8) for x in prob_a.tolist()],
        "prob_B": [round(x, 8) for x in prob_b.tolist()],
        "conf_A": round(conf_a, 4), "conf_B": round(conf_b, 4),
        "dev_prob_max": round(dev_prob, 8), "dev_conf": round(dev_conf, 6),
        "dev_logit_max": round(dev_logit, 7),
        "dev_gate_max": round(dev_gate, 7),
        "dev_img_feat_max": round(dev_feat_img, 6),
        "dev_txt_feat_max": round(dev_feat_txt, 6),
        "severity_A": SEVERITY_MAP[cls_a], "severity_B": SEVERITY_MAP[cls_b],
    })

summary = {
    "device": DEVICE,
    "n": len(samples),
    "checkpoint": str(GATED_CKPT),
    "max_dev_class": int(max_dev_class),
    "max_dev_prob": round(max_dev_prob, 8),
    "max_dev_conf": round(max_dev_conf, 6),
    "parity_pass": int(max_dev_class == 0 and max_dev_prob < 1e-6 and max_dev_conf < 1e-4),
}

out = {"summary": summary, "samples": results}
with open(os.path.join(ROOT, "research", "final_cleanup", "checkpoint_parity.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps(summary, indent=2))
print("saved research/final_cleanup/checkpoint_parity.json")