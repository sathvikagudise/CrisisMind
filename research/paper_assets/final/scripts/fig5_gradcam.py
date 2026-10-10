"""FIG 5 — REAL Grad-CAM examples from the FINAL leakage-free model.

Pipeline (identical to the frozen final evaluation):
  image : torchvision ImageNet EfficientNet-B0, classifier->Identity,
          Resize(224,224)+ToTensor+ImageNet-normalise   [FROZEN, no task training]
  text  : vanilla distilbert-base-uncased CLS embedding,
          URL-stripped tweet, max_len 64                [FROZEN]
  head  : GatedFusionWide v3 trunk
          backend/outputs/checkpoints/final_exp/F4_generic_wce_bal_n0.0_s42/best.pth

Explanations use the project's own Grad-CAM implementation
(backend/explainability/gradcam.py) targeting EfficientNet-B0 features.8,
wrapped so gradients flow from the FUSION logits back into the image tower.

Example selection is deterministic: first test rows (dataset order in
research/optimization/cleaned_dataset.csv) whose true class is correctly
predicted, one per severity class. Everything shown (GT / prediction /
confidence / heatmap) is computed here — nothing is fabricated.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import apply_style, save_all, ROOT
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from transformers import DistilBertModel, DistilBertTokenizerFast

from backend.explainability.gradcam import GradCAM
from backend.explainability.image_overlay import overlay_heatmap, heatmap_to_pil
from research.final_experiment.phase4_search import GatedFusionWide

DATASET = os.path.join(ROOT, "dataset_analysis", "CrisisMMD_v2.0")
SPLIT_CSV = os.path.join(ROOT, "research", "optimization", "cleaned_dataset.csv")
CKPT = os.path.join(ROOT, "backend", "outputs", "checkpoints", "final_exp",
                    "F4_generic_wce_bal_n0.0_s42", "best.pth")
IMG_SIZE, MAX_LEN = 224, 64
CLASS_NAMES = ["Little/No Damage", "Mild Damage", "Severe Damage"]

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_models():
    m = models.efficientnet_b0(weights=None)
    ref = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    sd = {k: v for k, v in ref.state_dict().items() if not k.startswith("classifier.")}
    m.load_state_dict(sd, strict=False)
    m.classifier = nn.Identity()
    m.eval()
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    bert.eval()
    fusion = GatedFusionWide(dropout=0.2)
    fusion.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))
    fusion.eval()
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return m, bert, fusion, tok


@torch.no_grad()
def txt_feature(bert, tok, raw_text):
    t = re.sub(r"http\S+|www\S+|https\S+", "", str(raw_text), flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip() or " "
    e = tok(t, truncation=True, padding="max_length", max_length=MAX_LEN,
            return_tensors="pt")
    out = bert(input_ids=e["input_ids"], attention_mask=e["attention_mask"])
    return out.last_hidden_state[:, 0, :]          # (1,768)


class FusionCamWrapper(nn.Module):
    """model(x) -> fusion logits; lets GradCAM target backbone.features.8."""
    def __init__(self, backbone, fusion, txt_feat):
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion
        self.register_buffer("txt_feat", txt_feat)
    def forward(self, x):
        img_feat = self.backbone(x)
        logits, _ = self.fusion(img_feat, self.txt_feat)
        return logits


def main():
    df = pd.read_csv(SPLIT_CSV)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"test rows: {len(test)}")

    img_enc, bert, fusion, tok = build_models()
    picked = {}                                    # label_int -> row idx
    for i, r in test.iterrows():
        y = int(r.label_int)
        if y in picked:
            continue
        try:
            im = Image.open(os.path.join(DATASET, r.image_path)).convert("RGB")
        except Exception:
            continue
        x = TF(im).unsqueeze(0)
        with torch.no_grad():
            tf_ = txt_feature(bert, tok, r.tweet_text)
            with torch.no_grad():
                feat = img_enc(x)
                logits, _ = fusion(feat, tf_)
            probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())
        if pred == y:                              # correct -> candidate
            picked[y] = dict(idx=i, path=r.image_path, gt=y, pred=pred,
                             conf=conf, pil=im.copy(), tensor=x,
                             txt=tf_, probs=probs.tolist())
            print(f"  class {y} ({CLASS_NAMES[y]}): row {i}, conf={conf*100:.1f}% "
                  f"-> {r.image_path}")
        if len(picked) == 3:
            break
    assert len(picked) == 3, f"only found correct predictions for {list(picked)}"

    # Real Grad-CAM through the fusion wrapper
    panels = []
    for y in sorted(picked):
        p = picked[y]
        wrapper = FusionCamWrapper(img_enc, fusion, p["txt"])
        cam = GradCAM(wrapper, target_layer="backbone.features.8")
        xin = p["tensor"].clone().requires_grad_(True)
        heatmap, cls_idx, logits = cam.generate_with_logits(xin, class_idx=None)
        pred = int(logits[0].argmax().item())
        conf = float(torch.softmax(logits, 1)[0, pred].item())
        assert pred == p["gt"], "CAM forward disagreed with selection pass"
        panels.append(dict(meta=p, heatmap=heatmap, pred=pred, conf=conf))

    # Figure: rows = examples, cols = Original | Heatmap | Overlay
    apply_style()
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(6.6, 2.35 * n))
    if n == 1:
        axes = axes[None, :]
    for r_i, panel in enumerate(panels):
        p, hm = panel["meta"], panel["heatmap"]
        orig = p["pil"].resize((IMG_SIZE, IMG_SIZE))
        heat = heatmap_to_pil(hm)
        over = overlay_heatmap(orig.copy(), hm, alpha=0.5)
        for c_i, (img, title) in enumerate([(orig, "Original"),
                                            (heat, "Grad-CAM heatmap"),
                                            (over, "Overlay")]):
            ax = axes[r_i, c_i]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r_i == 0:
                ax.set_title(title, fontsize=9.5)
            if c_i == 0:
                ax.set_ylabel(f"true: {CLASS_NAMES[p['gt']]}", fontsize=8.6)
            tag = "correct" if panel["pred"] == p["gt"] else "incorrect"
            color = "#1e8449" if tag == "correct" else "#b03a2e"
            ax.text(3, IMG_SIZE - 4, f"pred: {CLASS_NAMES[panel['pred']]}  "
                    f"({panel['conf']*100:.1f}%, {tag})",
                    fontsize=7.6, color="white", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.22", fc=color, ec="none", alpha=0.85),
                    transform=ax.transData)
    fig.tight_layout()
    paths = save_all(fig, "fig5_gradcam")

    # Persist the exact selections for provenance/QC
    rec = [{"true_class": CLASS_NAMES[p["meta"]["gt"]],
            "predicted_class": CLASS_NAMES[p["pred"]],
            "confidence_pct": round(p["conf"] * 100, 2),
            "image_path": p["meta"]["path"],
            "test_row_index": int(p["meta"]["idx"]),
            "target_layer": "EfficientNet-B0.features.8 (Conv2d)",
            "checkpoint": "final_exp/F4_generic_wce_bal_n0.0_s42/best.pth"}
           for p in panels]
    import json
    sel_path = os.path.join(os.path.dirname(paths["png"]), "fig5_gradcam_selections.json")
    json.dump(rec, open(sel_path, "w", encoding="utf-8"), indent=2)
    print("fig5 done; selections ->", os.path.relpath(sel_path, ROOT))


if __name__ == "__main__":
    main()
