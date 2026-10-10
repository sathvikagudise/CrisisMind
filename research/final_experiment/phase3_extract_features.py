"""Phase 3: Extract frozen features for ALL splits under two encoder arms.

Arms:
  dapt    : img = clean domain-adapted EfficientNet-B0 (final_exp/img_dapt.pth)
            txt = clean domain-adapted DistilBERT      (final_exp/txt_dapt.pth)
            (trained on TRAIN-ONLY data -> no leakage)
  generic : img = torchvision ImageNet EfficientNet-B0 (no task training)
            txt = HF distilbert-base-uncased vanilla CLS

Population: research/optimization/cleaned_dataset.csv (train 2209 / val 460 / test 436).
Test images are only passed through frozen encoders in eval mode (no training use).

Output: backend/outputs/checkpoints/final_exp/features_{arm}.pt
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import DistilBertModel, DistilBertTokenizerFast
from PIL import Image

DATASET = os.path.join(ROOT, "dataset_analysis", "CrisisMMD_v2.0")
SPLIT_CSV = os.path.join(ROOT, "research", "optimization", "cleaned_dataset.csv")
EXP_DIR = os.path.join(ROOT, "backend", "outputs", "checkpoints", "final_exp")
IMG_SIZE, TEXT_MODEL, MAX_LEN = 224, "distilbert-base-uncased", 64
TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class DS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tok = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            im = Image.open(os.path.join(DATASET, r.image_path)).convert("RGB")
        except Exception:
            im = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        import re
        t = re.sub(r"http\S+|www\S+|https\S+", "", str(r.tweet_text), flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t).strip()
        e = self.tok(t if t else " ", truncation=True, padding="max_length",
                     max_length=MAX_LEN, return_tensors="pt")
        return TF(im), e["input_ids"][0], e["attention_mask"][0], int(r.label_int)


def build_img_encoder(arm):
    m = efficientnet_b0(weights=None)
    if arm == "dapt":
        sd = torch.load(os.path.join(EXP_DIR, "img_dapt.pth"), map_location="cpu")
    else:  # generic ImageNet
        ref = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        sd = {k: v for k, v in ref.state_dict().items() if not k.startswith("classifier.")}
    m.load_state_dict(sd, strict=False)
    m.classifier = nn.Identity()
    m.eval()
    return m


def build_txt_encoder(arm):
    m = DistilBertModel.from_pretrained(TEXT_MODEL)
    if arm == "dapt":
        sd = torch.load(os.path.join(EXP_DIR, "txt_dapt.pth"), map_location="cpu")
        sub = {}
        for k, v in sd.items():
            if k.startswith("distilbert."):
                sub[k.replace("distilbert.", "")] = v
        m.load_state_dict(sub, strict=True)
    m.eval()
    return m


def extract(arm):
    out_path = os.path.join(EXP_DIR, f"features_{arm}.pt")
    if os.path.exists(out_path):
        print(f"cache exists: {out_path}"); return
    df = pd.read_csv(SPLIT_CSV)
    ds = DS(df)
    part_path = os.path.join(EXP_DIR, f"features_{arm}_partial.pt")
    imgs, txts, labels = [], [], []
    start = 0
    if os.path.exists(part_path):
        st = torch.load(part_path, map_location="cpu", weights_only=False)
        imgs, txts, labels, start = st["img"], st["txt"], st["labels"], st["count"]
        print(f"resuming {arm} at sample {start}/{len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    img_enc = build_img_encoder(arm)
    txt_enc = build_txt_encoder(arm)
    n = len(ds)
    t0 = time.time()
    with torch.no_grad():
        for bi, (im, ids, mask, y) in enumerate(loader):
            lo, hi = bi * 32, min((bi + 1) * 32, n)
            if hi <= start:
                continue
            if lo < start:
                im, ids, mask, y = im[start - lo:], ids[start - lo:], mask[start - lo:], y[start - lo:]
            imgs.append(img_enc(im))
            txts.append(txt_enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
            labels.append(y)
            done = hi
            if done % 640 == 0 or done == n:
                torch.save({"img": torch.cat(imgs), "txt": torch.cat(txts),
                            "labels": torch.cat(labels), "count": int(done)}, part_path)
                el = time.time() - t0
                print(f"  [{arm}] {done}/{n} ({el:.0f}s, eta {el/(done-start)*(n-done):.0f}s)", flush=True)
            if time.time() - t0 > 240:
                torch.save({"img": torch.cat(imgs), "txt": torch.cat(txts),
                            "labels": torch.cat(labels), "count": int(done)}, part_path)
                print(f"  [{arm}] BUDGET save at {done}/{n}; rerun", flush=True)
                sys.exit(2)
    idx = {sp: np.where(df["split"].values == sp)[0] for sp in ["train", "val", "test"]}
    torch.save({"img_feats": torch.cat(imgs), "txt_feats": torch.cat(txts),
                "labels": torch.cat(labels),
                "idx_train": idx["train"], "idx_val": idx["val"], "idx_test": idx["test"],
                "paths": df["image_path"].tolist(), "arm": arm}, out_path)
    if os.path.exists(part_path):
        os.remove(part_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    for arm in ["dapt", "generic"]:
        extract(arm)
