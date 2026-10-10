"""Manual evaluation: v1 vs v2 vs v3 on manual test images and text samples."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer

from backend.training.config import (
    LABEL_NAMES, NUM_CLASSES, FUSION_DROPOUT, CHECKPOINTS_DIR, IMG_SIZE,
    TEXT_MODEL_NAME, TEXT_MAX_LEN,
)
from backend.models.fusion_models import (
    GatedFusion, ImageFeatureExtractor, TextFeatureExtractor,
)
from backend.training.train_v3_batch2 import GatedFusionWide
from backend.config.shared import IMG_CKPT, TEXT_CKPT, DEVICE

DEV = torch.device(DEVICE)
V3_DIR = CHECKPOINTS_DIR / "v3"
RESULTS_DIR = Path("testing")
SEVERITY_LABELS = ["Little or No Damage", "Mild Damage", "Severe Damage"]

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_encoders():
    img_enc = ImageFeatureExtractor(IMG_CKPT, freeze=True).to(DEV).eval()
    txt_enc = TextFeatureExtractor(TEXT_CKPT, freeze=True).to(DEV).eval()
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    return img_enc, txt_enc, tokenizer


def extract_features(img_path, text, img_enc, txt_enc, tokenizer):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
    img_t = TF(img).unsqueeze(0).to(DEV)
    enc = tokenizer(str(text), truncation=True, padding="max_length",
                    max_length=TEXT_MAX_LEN, return_tensors="pt")
    ids = enc["input_ids"].to(DEV)
    mask = enc["attention_mask"].to(DEV)
    with torch.no_grad():
        img_f = img_enc(img_t)
        txt_f = txt_enc(ids, mask)
    return img_f, txt_f


def predict(model, img_f, txt_f):
    model.eval()
    with torch.no_grad():
        logits, gates = model(img_f, txt_f)
    pred = logits.argmax(1).item()
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    gate_vals = gates.cpu().numpy()[0]
    return pred, probs, gate_vals


def main():
    print("=" * 70)
    print("  CrisisMind Manual Evaluation — v1 vs v2 vs v3")
    print("=" * 70)

    img_enc, txt_enc, tokenizer = load_encoders()

    # Load models
    models = {}
    # v1
    m1 = GatedFusion(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES, dropout=FUSION_DROPOUT).to(DEV)
    m1.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_gated_fusion.pth", map_location=DEV))
    models["v1"] = m1
    # v2
    m2 = GatedFusion(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES, dropout=FUSION_DROPOUT).to(DEV)
    m2.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_gated_fusion_v2.pth", map_location=DEV))
    models["v2"] = m2
    # v3 (standard)
    m3s = GatedFusion(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES, dropout=FUSION_DROPOUT).to(DEV)
    v3_std = V3_DIR / "ls_sqrt_g0.5_lr1e-03_best.pth"
    if v3_std.exists():
        m3s.load_state_dict(torch.load(v3_std, map_location=DEV))
        models["v3_std"] = m3s
    # v3 (wide — the winner)
    m3w = GatedFusionWide(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES, dropout=0.3).to(DEV)
    m3w.load_state_dict(torch.load(V3_DIR / "wide_wce_g1.0_lr1e-04_best.pth", map_location=DEV))
    models["v3_wide"] = m3w

    print(f"  Loaded models: {list(models.keys())}\n")

    # ── Manual text samples ──
    text_samples = [
        ("The road has a few visible cracks but traffic is normal.", 0),
        ("Some tree branches fell on the lawn but the house is intact.", 0),
        ("Light rain caused minor puddles on the street.", 0),
        ("A few roof tiles shifted slightly during the storm.", 0),
        ("The fence has some minor scratches from wind-blown debris.", 0),
        ("Several buildings have broken windows and partial wall damage.", 1),
        ("Floodwater has entered the ground floor of the apartment complex.", 1),
        ("The church roof collapsed partially due to heavy snow load.", 1),
        ("Moderate cracks in the retaining wall along the highway.", 1),
        ("Three vehicles damaged by falling tree branches after the storm.", 1),
        ("Multiple buildings have collapsed and rescue teams are evacuating people.", 2),
        ("Entire neighbourhood destroyed by the wildfire leaving only ashes.", 2),
        ("The hospital has severe structural damage and is being evacuated.", 2),
        ("Hundreds of homes washed away by the catastrophic flood waters.", 2),
        ("Major earthquake has reduced the city center to complete rubble.", 2),
    ]

    print("─" * 70)
    print("  TEXT-ONLY EVALUATION")
    print("─" * 70)

    lines = []
    lines.append("# CrisisMind — Manual Evaluation\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
    lines.append("## Text-Only Evaluation\n")
    lines.append("| # | Text (truncated) | True | v1 Pred | v2 Pred | v3_w Pred | v1 OK? | v2 OK? | v3 OK? |")
    lines.append("|---|-----------------|------|---------|---------|-----------|--------|--------|--------|")

    v1_correct = v2_correct = v3_correct = 0
    for i, (text, true_lbl) in enumerate(text_samples, 1):
        # Use a dummy image for text-only (all zeros)
        dummy_img_f = torch.zeros(1, 1280).to(DEV)
        enc = tokenizer(text, truncation=True, padding="max_length",
                        max_length=TEXT_MAX_LEN, return_tensors="pt")
        ids = enc["input_ids"].to(DEV)
        mask = enc["attention_mask"].to(DEV)
        with torch.no_grad():
            txt_f = txt_enc(ids, mask)

        preds = {}
        for name, model in models.items():
            pred, probs, gates = predict(model, dummy_img_f, txt_f)
            preds[name] = (pred, probs, gates)

        true_name = SEVERITY_LABELS[true_lbl]
        v1p = preds["v1"][0]
        v2p = preds["v2"][0]
        v3p = preds["v3_wide"][0]
        v1_ok = "Y" if v1p == true_lbl else "N"
        v2_ok = "Y" if v2p == true_lbl else "N"
        v3_ok = "Y" if v3p == true_lbl else "N"
        if v1p == true_lbl: v1_correct += 1
        if v2p == true_lbl: v2_correct += 1
        if v3p == true_lbl: v3_correct += 1

        text_short = text[:40] + "..." if len(text) > 40 else text
        row = f"| {i} | {text_short} | {SEVERITY_LABELS[true_lbl][:3]} | {SEVERITY_LABELS[v1p][:3]} | {SEVERITY_LABELS[v2p][:3]} | {SEVERITY_LABELS[v3p][:3]} | {v1_ok} | {v2_ok} | {v3_ok} |"
        lines.append(row)
        print(f"  [{i:2d}] True={SEVERITY_LABELS[true_lbl][:3]}  v1={SEVERITY_LABELS[v1p][:3]}  v2={SEVERITY_LABELS[v2p][:3]}  v3w={SEVERITY_LABELS[v3p][:3]}")

    total = len(text_samples)
    lines.append(f"\n**Accuracy:** v1={v1_correct}/{total} ({100*v1_correct/total:.0f}%)  "
                 f"v2={v2_correct}/{total} ({100*v2_correct/total:.0f}%)  "
                 f"v3={v3_correct}/{total} ({100*v3_correct/total:.0f}%)")
    print(f"\n  Text-only: v1={v1_correct}/{total}  v2={v2_correct}/{total}  v3={v3_correct}/{total}")

    # ── Manual image evaluation ──
    manual_dir = Path("testing/manual_test")
    if manual_dir.exists():
        images = sorted(manual_dir.glob("*.jpg"))
        print(f"\n{'─'*70}")
        print(f"  IMAGE EVALUATION ({len(images)} images)")
        print(f"{'─'*70}")

        lines.append(f"\n## Image-Only Evaluation ({len(images)} images)\n")
        lines.append("| Image | v1 Pred | v1 ImgG | v2 Pred | v2 ImgG | v3 Pred | v3 ImgG |")
        lines.append("|-------|---------|---------|---------|---------|---------|---------|")

        for img_path in images:
            dummy_txt_f = torch.zeros(1, 768).to(DEV)
            img_f, _ = extract_features(str(img_path), "", img_enc, txt_enc, tokenizer)

            preds = {}
            for name, model in models.items():
                pred, probs, gates = predict(model, img_f, dummy_txt_f)
                preds[name] = (pred, probs, gates)

            fname = img_path.name
            v1p = preds["v1"][0]; v2p = preds["v2"][0]; v3p = preds["v3_wide"][0]
            v1g = preds["v1"][2][0]; v2g = preds["v2"][2][0]; v3g = preds["v3_wide"][2][0]

            lines.append(f"| {fname} | {SEVERITY_LABELS[v1p][:3]} | {v1g:.2f} | "
                         f"{SEVERITY_LABELS[v2p][:3]} | {v2g:.2f} | "
                         f"{SEVERITY_LABELS[v3p][:3]} | {v3g:.2f} |")
            print(f"  {fname}: v1={SEVERITY_LABELS[v1p][:3]}(I={v1g:.2f})  v2={SEVERITY_LABELS[v2p][:3]}(I={v2g:.2f})  v3w={SEVERITY_LABELS[v3p][:3]}(I={v3g:.2f})")

    # ── Multimodal evaluation ──
    print(f"\n{'─'*70}")
    print(f"  MULTIMODAL EVALUATION (image + text)")
    print(f"{'─'*70}")

    lines.append(f"\n## Multimodal Evaluation (image + text combined)\n")
    lines.append("| Image | True | v1 Pred | v2 Pred | v3 Pred | v1 Gate | v2 Gate | v3 Gate |")
    lines.append("|-------|------|---------|---------|---------|---------|---------|---------|")

    if manual_dir.exists():
        for img_path in images:
            # Extract ground truth from filename
            fname = img_path.name
            suffix = fname.split("_")[-1].replace(".jpg", "")
            true_lbl = int(suffix) if suffix in ("0", "1") else -1
            if true_lbl == -1:
                continue

            img_f, _ = extract_features(str(img_path), "", img_enc, txt_enc, tokenizer)
            # Use first text sample matching the label as representative text
            text = text_samples[true_lbl * 5][0]
            enc = tokenizer(text, truncation=True, padding="max_length",
                            max_length=TEXT_MAX_LEN, return_tensors="pt")
            ids = enc["input_ids"].to(DEV)
            mask = enc["attention_mask"].to(DEV)
            with torch.no_grad():
                txt_f = txt_enc(ids, mask)

            preds = {}
            for name, model in models.items():
                pred, probs, gates = predict(model, img_f, txt_f)
                preds[name] = (pred, probs, gates)

            v1p = preds["v1"][0]; v2p = preds["v2"][0]; v3p = preds["v3_wide"][0]
            v1g = preds["v1"][2]; v2g = preds["v2"][2]; v3g = preds["v3_wide"][2]

            lines.append(f"| {fname} | {SEVERITY_LABELS[true_lbl][:3]} | "
                         f"{SEVERITY_LABELS[v1p][:3]} | {SEVERITY_LABELS[v2p][:3]} | {SEVERITY_LABELS[v3p][:3]} | "
                         f"I={v1g[0]:.2f}T={v1g[1]:.2f} | I={v2g[0]:.2f}T={v2g[1]:.2f} | I={v3g[0]:.2f}T={v3g[1]:.2f} |")
            print(f"  {fname}: True={SEVERITY_LABELS[true_lbl][:3]}  v1={SEVERITY_LABELS[v1p][:3]}  v2={SEVERITY_LABELS[v2p][:3]}  v3w={SEVERITY_LABELS[v3p][:3]}")

    # ── Summary ──
    lines.append(f"\n## Summary\n")
    lines.append("| Metric | v1 | v2 | v3_wide |")
    lines.append("|--------|----|----|---------|")
    lines.append(f"| Text-only accuracy | {v1_correct}/{total} ({100*v1_correct/total:.0f}%) | "
                 f"{v2_correct}/{total} ({100*v2_correct/total:.0f}%) | "
                 f"{v3_correct}/{total} ({100*v3_correct/total:.0f}%) |")
    lines.append(f"| Gate collapse | Yes (T=0.014) | Fixed (T=0.431) | Fixed (T=0.436) |")
    lines.append(f"| Architecture | Standard | Standard | Wide (1024, BN) |")

    (RESULTS_DIR / "MANUAL_EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Saved MANUAL_EVALUATION.md")
    print(f"\n{'='*70}")
    print(f"  MANUAL EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
