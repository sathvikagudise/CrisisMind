"""Step 1: Extract and cache features from frozen encoders."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import time
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer
from backend.training.config import (
    RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    CHECKPOINTS_DIR, IMG_SIZE, TEXT_MODEL_NAME, TEXT_MAX_LEN,
)
from backend.datasets.multimodal_data import load_multimodal_data
from backend.models.fusion_models import ImageFeatureExtractor, TextFeatureExtractor
from backend.config.shared import IMG_CKPT, TEXT_CKPT, DEVICE
from sklearn.model_selection import train_test_split

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEV = torch.device(DEVICE)
V2_DIR = CHECKPOINTS_DIR / "v2"
V2_DIR.mkdir(parents=True, exist_ok=True)

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    print("=" * 50)
    print("  Step 1: Feature Extraction")
    print("=" * 50)

    cache_path = V2_DIR / "feature_cache.pt"
    if cache_path.exists():
        print(f"  Cache already exists at {cache_path}, skipping.")
        return

    # Also check v3 dir
    v3_cache = CHECKPOINTS_DIR / "v3" / "feature_cache.pt"
    if v3_cache.exists():
        import shutil
        V2_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v3_cache, cache_path)
        print(f"  Copied cache from {v3_cache} to {cache_path}, skipping extraction.")
        return

    print("Loading dataset...")
    df = load_multimodal_data()
    print(f"  Total: {len(df)} samples")

    y = df["label"].values
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y[train_idx])

    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    print("Loading frozen encoders...")
    img_enc = ImageFeatureExtractor(IMG_CKPT, freeze=True).to(DEV).eval()
    txt_enc = TextFeatureExtractor(TEXT_CKPT, freeze=True).to(DEV).eval()
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)

    n = len(df)
    img_feats = []
    txt_feats = []
    labels = []
    t0 = time.time()

    print(f"Extracting features for {n} samples on {DEVICE}...")
    with torch.no_grad():
        for i in range(n):
            row = df.iloc[i]
            try:
                img = Image.open(row["full_image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)

            img_t = TF(img).unsqueeze(0).to(DEV)
            enc = tokenizer(str(row["tweet_text"]), truncation=True, padding="max_length",
                            max_length=TEXT_MAX_LEN, return_tensors="pt")
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
    }, cache_path)

    # Also save to v3 directory
    v3_cache = CHECKPOINTS_DIR / "v3"
    v3_cache.mkdir(parents=True, exist_ok=True)
    torch.save({
        "img_feats": img_feats,
        "txt_feats": txt_feats,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }, v3_cache / "feature_cache.pt")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({n/elapsed:.1f} samples/sec)")
    print(f"  img_feats: {img_feats.shape}")
    print(f"  txt_feats: {txt_feats.shape}")
    print(f"  Saved to: {cache_path}")


if __name__ == "__main__":
    main()
