"""Full test set evaluation with confusion matrix for all 3 models."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import csv
import time
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from backend.analysis_engine.model_loader import load_all_models
from backend.api.preprocessing import preprocess_image, tokenize_text
from backend.api.config import SEVERITY_MAP, DEVICE
from backend.config.shared import PLOTS_DIR, METRICS_DIR
from backend.training.config import LABEL_NAMES, IMG_SIZE
from backend.datasets.multimodal_data import load_multimodal_data
from torchvision import transforms
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from backend.training.config import RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT

VAL_AUG = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SEVERITY_LABELS = ["Little or No Damage", "Mild Damage", "Severe Damage"]


def run():
    print("=" * 80)
    print("  Full Test Set Evaluation")
    print("=" * 80)

    # Load data
    print("\nLoading dataset...")
    df = load_multimodal_data()
    y = df["label"].values
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y[train_idx]
    )
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    print(f"Test set: {len(test_df)} samples")
    dist = test_df["label"].value_counts().sort_index()
    for label, count in dist.items():
        print(f"  {LABEL_NAMES[label]:30s}  {count}  ({count/len(test_df)*100:.1f}%)")

    # Load models
    print("\nLoading models...")
    models = load_all_models()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    all_preds = {"image_only": [], "text_only": [], "multimodal": []}
    all_labels = []
    all_confidences = {"image_only": [], "text_only": [], "multimodal": []}
    all_raw_probs = {"image_only": [], "text_only": [], "multimodal": []}

    print(f"\nRunning inference on {len(test_df)} samples...")
    t0 = time.time()

    for idx, row in test_df.iterrows():
        img_path = row["full_image_path"]
        text = str(row["tweet_text"])
        label = int(row["label"])
        all_labels.append(label)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        img_tensor = VAL_AUG(img).unsqueeze(0).to(DEVICE)
        encoded = tokenizer(text, truncation=True, padding="max_length",
                           max_length=64, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attn_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            # Image-only
            img_logits = models["image_classifier"](img_tensor)
            img_probs = torch.softmax(img_logits, dim=1).cpu().numpy()[0]

            # Text-only
            txt_out = models["text_classifier"](input_ids, attn_mask)
            txt_logits = txt_out.logits if hasattr(txt_out, "logits") else txt_out
            txt_probs = torch.softmax(txt_logits, dim=1).cpu().numpy()[0]

            # Multimodal
            img_feat = models["img_encoder"](img_tensor)
            txt_feat = models["txt_encoder"](input_ids, attn_mask)
            mm_logits, gates = models["gated_model"](img_feat, txt_feat)
            mm_probs = torch.softmax(mm_logits, dim=1).cpu().numpy()[0]

        for mode, probs in [("image_only", img_probs), ("text_only", txt_probs), ("multimodal", mm_probs)]:
            pred = int(np.argmax(probs))
            all_preds[mode].append(pred)
            all_confidences[mode].append(float(probs[pred] * 100))
            all_raw_probs[mode].append(probs.tolist())

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {idx+1}/{len(test_df)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Evaluate each mode
    y_true = np.array(all_labels)

    for mode in ["image_only", "text_only", "multimodal"]:
        y_pred = np.array(all_preds[mode])
        confs = np.array(all_confidences[mode])

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        f1w = f1_score(y_true, y_pred, average="weighted")
        pm = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rm = recall_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"\n{'=' * 60}")
        print(f"  {mode.upper()}")
        print(f"{'=' * 60}")
        print(f"  Accuracy:           {acc:.4f}")
        print(f"  F1 (macro):         {f1m:.4f}")
        print(f"  F1 (weighted):      {f1w:.4f}")
        print(f"  Precision (macro):  {pm:.4f}")
        print(f"  Recall (macro):     {rm:.4f}")
        print(f"  Avg confidence:     {confs.mean():.1f}%")
        print()

        report = classification_report(y_true, y_pred,
                                       target_names=SEVERITY_LABELS,
                                       zero_division=0)
        print(report)

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(f"                    Pred Little  Pred Mild  Pred Severe")
        for i, name in enumerate(SEVERITY_LABELS):
            print(f"  Actual {name:20s}  {cm[i][0]:6d}     {cm[i][1]:5d}      {cm[i][2]:6d}")

        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=ax)
        ax.set_title(f"CrisisMind — {mode.replace('_', ' ').title()} Confusion Matrix",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"confusion_matrix_{mode}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {PLOTS_DIR / f'confusion_matrix_{mode}.png'}")

        # Per-class recall
        for i, name in enumerate(SEVERITY_LABELS):
            total = cm[i].sum()
            correct = cm[i][i]
            print(f"  Recall {name:20s}: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else f"  Recall {name}: 0/0")

    # Prediction distribution analysis
    print(f"\n{'=' * 60}")
    print(f"  PREDICTION DISTRIBUTION ANALYSIS")
    print(f"{'=' * 60}")
    for mode in ["image_only", "text_only", "multimodal"]:
        preds = all_preds[mode]
        counts = {i: preds.count(i) for i in range(3)}
        total = len(preds)
        print(f"\n  {mode}:")
        for i in range(3):
            print(f"    {SEVERITY_LABELS[i]:25s}: {counts[i]:4d}/{total}  ({counts[i]/total*100:.1f}%)")

    # Per-class confidence breakdown
    print(f"\n{'=' * 60}")
    print(f"  PER-CLASS CONFIDENCE (mean ± std)")
    print(f"{'=' * 60}")
    for mode in ["image_only", "text_only", "multimodal"]:
        print(f"\n  {mode}:")
        for cls in range(3):
            cls_mask = y_true == cls
            if cls_mask.sum() > 0:
                cls_confs = confs[cls_mask] if mode == "multimodal" else np.array(all_confidences[mode])[cls_mask]
                print(f"    True={SEVERITY_LABELS[cls]:25s}: {cls_confs.mean():.1f}% ± {cls_confs.std():.1f}%  (n={cls_mask.sum()})")


if __name__ == "__main__":
    run()
