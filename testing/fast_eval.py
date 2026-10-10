"""Fast evaluation: sample 100 from test set per mode."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import time
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from backend.analysis_engine.model_loader import load_all_models
from backend.api.preprocessing import preprocess_image, tokenize_text
from backend.api.config import SEVERITY_MAP, DEVICE
from backend.config.shared import PLOTS_DIR
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
    print("  Fast Evaluation (all test samples)")
    print("=" * 80)

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
    for label in range(3):
        n = (test_df["label"] == label).sum()
        print(f"  {LABEL_NAMES[label]:30s}  {n}  ({n/len(test_df)*100:.1f}%)")

    models = load_all_models()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    results = {"image_only": [], "text_only": [], "multimodal": []}
    all_labels = []

    print(f"\nRunning inference on {len(test_df)} samples...")
    t0 = time.time()
    
    for i, row in test_df.iterrows():
        img_path = row["full_image_path"]
        text = str(row["tweet_text"])
        label = int(row["label"])
        all_labels.append(label)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)

        img_tensor = VAL_AUG(img).unsqueeze(0).to(DEVICE)
        encoded = tokenizer(text, truncation=True, padding="max_length",
                           max_length=64, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attn_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            img_logits = models["image_classifier"](img_tensor)
            img_pred = int(torch.argmax(img_logits, dim=1).item())

            txt_out = models["text_classifier"](input_ids, attn_mask)
            txt_logits = txt_out.logits if hasattr(txt_out, "logits") else txt_out
            txt_pred = int(torch.argmax(txt_logits, dim=1).item())

            img_feat = models["img_encoder"](img_tensor)
            txt_feat = models["txt_encoder"](input_ids, attn_mask)
            mm_logits, gates = models["gated_model"](img_feat, txt_feat)
            mm_pred = int(torch.argmax(mm_logits, dim=1).item())

        results["image_only"].append(img_pred)
        results["text_only"].append(txt_pred)
        results["multimodal"].append(mm_pred)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_df)} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(test_df)/elapsed:.1f} samples/sec)")

    y_true = np.array(all_labels)

    for mode in ["image_only", "text_only", "multimodal"]:
        y_pred = np.array(results[mode])

        print(f"\n{'='*60}")
        print(f"  {mode.upper()}")
        print(f"{'='*60}")

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        f1w = f1_score(y_true, y_pred, average="weighted")

        print(f"  Accuracy:    {acc:.4f}")
        print(f"  F1 (macro):  {f1m:.4f}")
        print(f"  F1 (weighted): {f1w:.4f}")

        report = classification_report(y_true, y_pred, target_names=SEVERITY_LABELS, zero_division=0)
        print(report)

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(f"  {'':25s}  Pred Little  Pred Mild  Pred Severe")
        for i, name in enumerate(SEVERITY_LABELS):
            print(f"  Actual {name:20s}  {cm[i][0]:6d}     {cm[i][1]:5d}      {cm[i][2]:6d}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=ax)
        ax.set_title(f"CrisisMind - {mode.replace('_',' ').title()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"confusion_matrix_{mode}.png", dpi=150)
        plt.close(fig)

    # Summary
    print(f"\n{'='*60}")
    print(f"  PREDICTION DISTRIBUTION")
    print(f"{'='*60}")
    for mode in ["image_only", "text_only", "multimodal"]:
        preds = results[mode]
        counts = [preds.count(i) for i in range(3)]
        print(f"  {mode:15s}: Little={counts[0]:4d}  Mild={counts[1]:4d}  Severe={counts[2]:4d}")
    
    # Per-class accuracy
    print(f"\n{'='*60}")
    print(f"  PER-CLASS ACCURACY")
    print(f"{'='*60}")
    for mode in ["image_only", "text_only", "multimodal"]:
        y_pred = np.array(results[mode])
        print(f"\n  {mode}:")
        for cls in range(3):
            mask = y_true == cls
            if mask.sum() > 0:
                correct = (y_pred[mask] == cls).sum()
                print(f"    {SEVERITY_LABELS[cls]:25s}: {correct}/{mask.sum()} = {correct/mask.sum()*100:.1f}%")


if __name__ == "__main__":
    run()
