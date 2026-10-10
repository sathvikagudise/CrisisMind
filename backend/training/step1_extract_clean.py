"""Step 1 (cleaned): Extract and cache features for the cleaned CrisisMMD subset.

Replicates backend/training/step1_extract.py exactly (same transforms, tokenizer,
frozen encoders, text cleaning, output dtype) but over the cleaned population defined
in research/data_cleaning/KEPT_SAMPLES.csv, using the leak-free split from Phase 4.

Output: backend/outputs/checkpoints/v3_clean/feature_cache.pt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer

from backend.training.config import (
    RANDOM_SEED, CHECKPOINTS_DIR, IMG_SIZE, TEXT_MODEL_NAME, TEXT_MAX_LEN,
)
from backend.models.fusion_models import ImageFeatureExtractor, TextFeatureExtractor
from backend.config.shared import IMG_CKPT, TEXT_CKPT, DEVICE
from backend.datasets.multimodal_data import clean_tweet

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEV = torch.device(DEVICE)
ROOT = Path(__file__).resolve().parent.parent.parent
KEPT_CSV = ROOT / "research" / "data_cleaning" / "KEPT_SAMPLES.csv"
OUT_DIR = CHECKPOINTS_DIR / "v3_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUT_DIR / "feature_cache.pt"

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    print("=" * 60)
    print("  Step 1 (clean): Feature Extraction on cleaned population")
    print("=" * 60)

    if CACHE_PATH.exists():
        print(f"  Cache already exists at {CACHE_PATH}, skipping.")
        return

    kept = pd.read_csv(KEPT_CSV)
    kept["full_image_path"] = kept["image_path"].apply(
        lambda p: str((ROOT / "dataset_analysis" / "CrisisMMD_v2.0" / p).resolve())
    )
    kept["clean_text"] = kept["tweet_text"].apply(clean_tweet)
    print(f"  Cleaned population: {len(kept)} samples")

    split_map = {"train": 0, "val": 1, "test": 2}
    kept["split_id"] = kept["split"].map(split_map)
    n = len(kept)
    indices = np.arange(n)
    train_idx = indices[kept["split_id"].values == 0]
    val_idx = indices[kept["split_id"].values == 1]
    test_idx = indices[kept["split_id"].values == 2]
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    print("Loading frozen encoders...")
    img_enc = ImageFeatureExtractor(IMG_CKPT, freeze=True).to(DEV).eval()
    txt_enc = TextFeatureExtractor(TEXT_CKPT, freeze=True).to(DEV).eval()
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)

    img_feats, txt_feats, labels = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for i in range(n):
            row = kept.iloc[i]
            try:
                img = Image.open(row["full_image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)

            img_t = TF(img).unsqueeze(0).to(DEV)
            enc = tokenizer(str(row["clean_text"]), truncation=True,
                            padding="max_length", max_length=TEXT_MAX_LEN, return_tensors="pt")
            ids = enc["input_ids"].to(DEV)
            mask = enc["attention_mask"].to(DEV)

            img_feats.append(img_enc(img_t).cpu())
            txt_feats.append(txt_enc(ids, mask).cpu())
            labels.append(int(row["label"]))

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  {i+1}/{n} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    img_feats = torch.cat(img_feats)
    txt_feats = torch.cat(txt_feats)
    labels = torch.tensor(labels, dtype=torch.long)

    torch.save({
        "img_feats": img_feats,
        "txt_feats": txt_feats,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "paths": kept["image_path"].tolist(),
        "population": "CrisisMMD_v2.0 cleaned subset (n=3105)",
        "split": "leak-free stratified 70/15/15 (Phase 4)",
    }, CACHE_PATH)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({n/elapsed:.1f} samples/sec)")
    print(f"  img_feats: {img_feats.shape}  txt_feats: {txt_feats.shape}  labels: {labels.shape}")
    print(f"  Saved to: {CACHE_PATH}")


if __name__ == "__main__":
    main()
