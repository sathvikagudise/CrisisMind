"""Phase 2: Retrain CLEAN encoders on TRAIN-ONLY data (removes inherited leakage).

Motivation (documented integrity finding):
  The existing best_model.pth / best_text_model.pth were trained on the ORIGINAL
  seed-42 split of the 3,526 population. 308/436 images of the current test set were
  in those encoders' training set -> every prior frozen-feature metric carries
  indirect test leakage. Valid science requires encoders that never saw val/test.

This script trains:
  --task img : EfficientNet-B0 init=ImageNet, supervised severity on EXPANDED train
               (2,209 kept-train + 312 safe reclaimed rule-H samples = 2,521),
               moderate augmentation, WCE(inv-freq), early stop on VAL macro-F1.
  --task txt : DistilBERT init=distilbert-base-uncased (fresh, un-leaked),
               severity classification on same train-only texts, WCE, lr 2e-5.

Val/test are NEVER trained on. Test is never touched anywhere.
Outputs -> backend/outputs/checkpoints/final_exp/{img_dapt.pth, txt_dapt.pth}
"""
import sys, os, json, time, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import DistilBertModel, DistilBertTokenizerFast
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

OUT_DIR = os.path.join(ROOT, "backend", "outputs", "checkpoints", "final_exp")
os.makedirs(OUT_DIR, exist_ok=True)
DATASET = os.path.join(ROOT, "dataset_analysis", "CrisisMMD_v2.0")
RECLAIM = os.path.join(ROOT, "research", "final_experiment", "RECLAIM_TRAIN_ONLY.csv")
OPT_SPLIT = os.path.join(ROOT, "research", "optimization", "cleaned_dataset.csv")
IMG_SIZE, TEXT_MODEL, MAX_LEN, N_CLS, DEV = 224, "distilbert-base-uncased", 64, 3, "cpu"
SEED = 42


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)


def load_train_frame():
    df = pd.read_csv(OPT_SPLIT)
    tr = df[df.split == "train"][["image_path", "tweet_text", "label_int"]].copy()
    rc = pd.read_csv(RECLAIM)
    rc = rc.rename(columns={"original_label": "image_damage", "text": "tweet_text"})
    lm = {"little_or_no_damage": 0, "mild_damage": 1, "severe_damage": 2}
    rc["label_int"] = rc["image_damage"].map(lm)
    rc = rc[["image_path", "tweet_text", "label_int"]]
    tr = pd.concat([tr, rc], ignore_index=True)
    assert tr.label_int.notna().all() and len(tr) == len(df[df.split == "train"]) + len(rc)
    return tr.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def load_val_frame():
    df = pd.read_csv(OPT_SPLIT)
    return df[df.split == "val"][["image_path", "tweet_text", "label_int"]].reset_index(drop=True)


class ImgDS(Dataset):
    def __init__(self, df, tf=None):
        self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            im = Image.open(os.path.join(DATASET, r.image_path)).convert("RGB")
        except Exception:
            im = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return (self.tf(im) if self.tf else self.tf_eval(im)), int(r.label_int)

    @staticmethod
    def tf_eval(im):
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(im)


class TxtDS(Dataset):
    def __init__(self, df, tok):
        self.df = df.reset_index(drop=True); self.tok = tok
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        t = str(r.tweet_text)
        t = __import__("re").sub(r"http\S+|www\S+|https\S+", "", t, flags=__import__("re").IGNORECASE)
        t = __import__("re").sub(r"\s+", " ", t).strip()
        e = self.tok(t, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        return e["input_ids"][0], e["attention_mask"][0], int(r.label_int)


def wce_weights(labels):
    cnt = np.bincount(labels, minlength=N_CLS).astype(np.float64)
    w = 1.0 / cnt; w = w / w.sum() * N_CLS
    return torch.tensor(w, dtype=torch.float32)


def evaluate_img(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x).argmax(1)
            ys += y.tolist(); ps += p.tolist()
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def train_img(df_tr, df_va, budget_sec=1e9):
    if os.path.exists(os.path.join(OUT_DIR, "img_dapt.pth")):
        print("  img_dapt.pth already exists - skipping"); return
    seed_all(SEED)
    tf_tr = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr_ds = ImgDS(df_tr, tf=tf_tr); va_ds = ImgDS(df_va, tf=lambda im: ImgDS.tf_eval(im))
    w = wce_weights(df_tr.label_int.values)
    crit = nn.CrossEntropyLoss(weight=w)
    sw = (1.0 / np.bincount(df_tr.label_int.values, minlength=N_CLS))[df_tr.label_int.values].tolist()
    tr_loader = DataLoader(tr_ds, batch_size=32, sampler=WeightedRandomSampler(sw, len(sw), True), num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, N_CLS)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)

    MAX_EP, PAT = 8, 3
    rs_path = os.path.join(OUT_DIR, "img_resume.pt")
    ep0, best_f1, best_ep, bad = 1, 0.0, 0, 0
    t_start = time.time()
    if os.path.exists(rs_path):
        st = torch.load(rs_path, map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        ep0, best_f1, best_ep, bad = st["epoch"] + 1, st["best_f1"], st["best_ep"], st["bad"]
        print(f"  resumed at epoch {ep0} (best_f1={best_f1:.4f})", flush=True)

    t0 = time.time()
    done = True
    for ep in range(ep0, MAX_EP + 1):
        model.train(); tot = 0; n = 0; te = time.time()
        for x, y in tr_loader:
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        acc, f1m = evaluate_img(model, va_loader)
        marker = ""
        if f1m > best_f1:
            best_f1, best_ep, bad = f1m, ep, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "img_dapt_best.pth")); marker = " *"
        else:
            bad += 1
        print(f"  [img] ep{ep:02d} loss={tot/n:.4f} val_acc={acc:.4f} val_F1m={f1m:.4f} ({time.time()-te:.0f}s){marker}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best_f1": best_f1,
                    "best_ep": best_ep, "bad": bad}, rs_path)
        if bad >= PAT:
            print("  early stop"); break
        if time.time() - t_start > budget_sec:
            print(f"  BUDGET reached after ep{ep}; rerun to continue"); done = False; break

    if not done:
        sys.exit(2)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "img_dapt_best.pth"), map_location="cpu"))
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("classifier.")}
    torch.save(sd, os.path.join(OUT_DIR, "img_dapt.pth"))
    os.remove(rs_path)
    json.dump({"best_epoch": best_ep, "val_f1_macro": best_f1,
               "minutes": round((time.time()-t0)/60, 1)},
              open(os.path.join(OUT_DIR, "img_dapt_meta.json"), "w"), indent=2)
    print(f"  IMG DONE best_val_F1m={best_f1:.4f}", flush=True)


def evaluate_txt(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            p = model(input_ids=ids, attention_mask=mask).argmax(1)
            ys += y.tolist(); ps += p.tolist()
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


class TxtCls(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(TEXT_MODEL)
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, N_CLS)
    def forward(self, input_ids, attention_mask):
        h = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.classifier(self.dropout(nn.ReLU()(self.pre_classifier(h))))


def train_txt(df_tr, df_va, budget_sec=1e9):
    if os.path.exists(os.path.join(OUT_DIR, "txt_dapt.pth")):
        print("  txt_dapt.pth already exists - skipping"); return
    seed_all(SEED)
    tok = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL)
    tr_ds, va_ds = TxtDS(df_tr, tok), TxtDS(df_va, tok)
    w = wce_weights(df_tr.label_int.values)
    crit = nn.CrossEntropyLoss(weight=w)
    sw = (1.0 / np.bincount(df_tr.label_int.values, minlength=N_CLS))[df_tr.label_int.values].tolist()
    tr_loader = DataLoader(tr_ds, batch_size=16, sampler=WeightedRandomSampler(sw, len(sw), True), num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    model = TxtCls()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)

    MAX_EP, PAT = 5, 2
    rs_path = os.path.join(OUT_DIR, "txt_resume.pt")
    ep0, best_f1, best_ep, bad = 1, 0.0, 0, 0
    t_start = time.time()
    if os.path.exists(rs_path):
        st = torch.load(rs_path, map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        ep0, best_f1, best_ep, bad = st["epoch"] + 1, st["best_f1"], st["best_ep"], st["bad"]
        print(f"  resumed at epoch {ep0} (best_f1={best_f1:.4f})", flush=True)

    t0 = time.time(); done = True
    for ep in range(ep0, MAX_EP + 1):
        model.train(); tot = 0; n = 0; te = time.time()
        for ids, mask, y in tr_loader:
            loss = crit(model(input_ids=ids, attention_mask=mask), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * ids.size(0); n += ids.size(0)
        acc, f1m = evaluate_txt(model, va_loader)
        marker = ""
        if f1m > best_f1:
            best_f1, best_ep, bad = f1m, ep, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "txt_dapt_best.pth")); marker = " *"
        else:
            bad += 1
        print(f"  [txt] ep{ep:02d} loss={tot/n:.4f} val_acc={acc:.4f} val_F1m={f1m:.4f} ({time.time()-te:.0f}s){marker}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best_f1": best_f1, "best_ep": best_ep, "bad": bad}, rs_path)
        if bad >= PAT:
            print("  early stop"); break
        if time.time() - t_start > budget_sec:
            print(f"  BUDGET reached after ep{ep}; rerun to continue"); done = False; break

    if not done:
        sys.exit(2)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "txt_dapt_best.pth"), map_location="cpu"))
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "txt_dapt.pth"))
    os.remove(rs_path)
    json.dump({"best_epoch": best_ep, "val_f1_macro": best_f1,
               "minutes": round((time.time()-t0)/60, 1)},
              open(os.path.join(OUT_DIR, "txt_dapt_meta.json"), "w"), indent=2)
    print(f"  TXT DONE best_val_F1m={best_f1:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["img", "txt"], required=True)
    ap.add_argument("--budget", type=float, default=1e9, help="seconds of epoch budget before exit(2)")
    a = ap.parse_args()
    tr = load_train_frame()
    va = load_val_frame()
    print(f"TRAIN(expanded)={len(tr)}  VAL={len(va)}  (test never loaded here)")
    print(tr.label_int.value_counts().to_dict())
    if a.task == "img":
        train_img(tr, va, budget_sec=a.budget)
    else:
        train_txt(tr, va, budget_sec=a.budget)
