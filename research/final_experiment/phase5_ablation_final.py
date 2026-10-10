"""Phase 5: Multimodal ablation study + FROZEN final test evaluation.

Ablations (all trained with the SELECTED recipe, seed 42, on cached features):
  image-only | text-only | concat | gated-v1 | gated-v2 | gated-v3(selected)

The FINAL model is the selected config's checkpoint (chosen purely by validation
macro-F1 across the multi-seed runs). Its test evaluation happens here exactly once,
with full metrics, confusion matrix, per-class report, and inference latency
(fusion-head-only and full raw-input pipeline).

Outputs: backend/outputs/checkpoints/final_exp/ablation_results.json
         backend/outputs/checkpoints/final_exp/FINAL_TEST_RESULTS.json
"""
import sys, os, json, time, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, confusion_matrix)

EXP_DIR = os.path.join(ROOT, "backend", "outputs", "checkpoints", "final_exp")
DATASET = os.path.join(ROOT, "dataset_analysis", "CrisisMMD_v2.0")
N_CLS = 3
LABEL_NAMES = ["little_or_no_damage", "mild_damage", "severe_damage"]
IMG_SIZE = 224

# import shared pieces from phase4
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_search import (GatedFusionWide, coral_predict, make_loss, FeatDS,
                           predict, metrics_dict)


class UnimodalHead(nn.Module):
    """Same wide-trunk style, single modality (fair unimodal ablation).
    forward(im, tx) selects the modality internally."""
    def __init__(self, dim, mode, dropout=0.2):
        super().__init__()
        self.mode = mode
        self.fc1 = nn.Linear(dim, 1024); self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512); self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, N_CLS)
    def forward(self, im, tx=None):
        x = im if self.mode == "img" else tx
        x = self.bn1(self.fc1(x)); x = self.relu(x); x = self.dropout(x)
        x = self.bn2(self.fc2(x)); x = self.relu(x); x = self.dropout(x)
        return self.fc3(x), torch.zeros(len(x), 2)


class ConcatFusion(nn.Module):
    def __init__(self, img_dim=1280, txt_dim=768, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(img_dim + txt_dim, 512)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, N_CLS)
    def forward(self, im, tx):
        x = self.fc1(torch.cat([im, tx], 1))
        return self.fc2(self.dropout(self.relu(x))), torch.zeros(len(x), 2)


class GatedFusionV1(nn.Module):
    """v1: narrow gate net + 2-layer head."""
    def __init__(self, img_dim=1280, txt_dim=768, dropout=0.3):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + txt_dim, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid())
        self.fc1 = nn.Linear(img_dim + txt_dim, 512)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, N_CLS)
    def forward(self, im, tx):
        concat = torch.cat([im, tx], 1)
        gates = self.gate_net(concat)
        fused = torch.cat([gates[:, 0:1] * im, gates[:, 1:2] * tx], 1)
        x = self.fc1(fused)
        return self.fc2(self.dropout(self.relu(x))), gates


class GatedFusionV2(GatedFusionV1):
    """v2 = v1 arch + gate regularization (target 0.5, w 0.01) + entropy bonus (0.005),
    matching backend/training/train_v2.py."""
    pass


def gate_reg(gates, w=0.01):
    return w * torch.mean((gates - 0.5) ** 2)


def gate_entropy(gates, w=0.005, eps=1e-8):
    ent = -(gates * torch.log(gates + eps) + (1 - gates) * torch.log(1 - gates + eps))
    return -w * ent.mean()   # maximize entropy -> minimize negative entropy


def train_abl(name, model_kind, cache, loss_name, sampling, noise, seed=42,
              lr=1e-4, dropout=0.2, max_epochs=30, patience=8):
    torch.manual_seed(seed); np.random.seed(seed)
    im, tx, y = cache["img_feats"], cache["txt_feats"], cache["labels"]
    itr, iv, it = cache["idx_train"], cache["idx_val"], cache["idx_test"]
    tr_y = y[itr].numpy()
    counts = np.bincount(tr_y, minlength=N_CLS).astype(np.float64)
    inv_w = (1.0 / counts); inv_w = inv_w / inv_w.sum() * N_CLS
    inv_t = torch.tensor(inv_w, dtype=torch.float32)
    criterion, _ = make_loss(loss_name, inv_t, counts)

    if model_kind == "img":
        model = UnimodalHead(1280, "img", dropout)
    elif model_kind == "txt":
        model = UnimodalHead(768, "txt", dropout)
    else:
        model = {"concat": lambda: ConcatFusion(dropout=dropout),
                 "v1": lambda: GatedFusionV1(dropout=dropout),
                 "v2": lambda: GatedFusionV2(dropout=dropout)}[model_kind]()
    fwd = lambda a, b: model(a, b)

    sw = (1.0 / counts)[tr_y].tolist()
    if sampling == "bal":
        sampler = WeightedRandomSampler(sw, num_samples=len(tr_y), replacement=True)
        tr_loader = DataLoader(FeatDS(im[itr], tx[itr], y[itr]), batch_size=32,
                               sampler=sampler, drop_last=True)
    else:
        g = torch.Generator(); g.manual_seed(seed)
        tr_loader = DataLoader(FeatDS(im[itr], tx[itr], y[itr]), batch_size=32,
                               shuffle=True, generator=g, drop_last=True)
    va_loader = DataLoader(FeatDS(im[iv], tx[iv], y[iv]), batch_size=64)
    te_loader = DataLoader(FeatDS(im[it], tx[it], y[it]), batch_size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 3
    def lrl(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(1, max_epochs - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lrl)

    best_val, bad = -1, 0
    out_dir = os.path.join(EXP_DIR, f"ABL_{name}")
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(1, max_epochs + 1):
        model.train(); tot = 0; nseen = 0
        for a, b, yb in tr_loader:
            if noise > 0:
                a = a + torch.randn_like(a) * noise
                b = b + torch.randn_like(b) * noise
            z, gates = fwd(a, b)
            loss = criterion(z, yb)
            if model_kind == "v2":
                loss = loss + gate_reg(gates) + gate_entropy(gates)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * a.size(0); nseen += a.size(0)
        sched.step()
        vy, vp, _ = predict(model, va_loader, "softmax") if not hasattr(model, "head_kind") \
            else predict(model, va_loader, model.head_kind)
        vf1 = f1_score(vy, vp, average="macro")
        if vf1 > best_val:
            best_val = vf1; bad = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth"), map_location="cpu"))
    vy, vp, _ = predict(model, va_loader, "softmax")
    ty, tp, _ = predict(model, te_loader, "softmax")
    res = {"name": name, "kind": model_kind, "val": metrics_dict(vy, vp),
           "test": metrics_dict(ty, tp)}
    print(f"  {name}: ValF1m={res['val']['f1_macro']:.4f} TestF1m={res['test']['f1_macro']:.4f}", flush=True)
    return res


# ─────────────────────────────────────────────────────────────────────
def full_pipeline_latency(checkpoint_dir, df_test, n_samples=60):
    """Average ms/sample for raw file -> prediction (B0 + DistilBERT + fusion head)."""
    from torchvision import transforms
    from torchvision.models import efficientnet_b0
    from transformers import DistilBertModel, DistilBertTokenizerFast
    import re
    from PIL import Image

    TF = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_enc = efficientnet_b0(weights=None)
    sd = torch.load(os.path.join(EXP_DIR, "img_dapt.pth"), map_location="cpu")
    img_enc.load_state_dict(sd, strict=False)
    img_enc.classifier = nn.Identity(); img_enc.eval()

    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tsd = torch.load(os.path.join(EXP_DIR, "txt_dapt.pth"), map_location="cpu")
    bert.load_state_dict({k.replace("distilbert.", ""): v for k, v in tsd.items()
                          if k.startswith("distilbert.")}, strict=True)
    bert.eval()
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # fusion head: infer dims from state dict
    fstate = torch.load(os.path.join(checkpoint_dir, "best.pth"), map_location="cpu")
    head = GatedFusionWide(dropout=0.2)
    head.load_state_dict(fstate); head.eval()

    times = []
    sub = df_test.head(n_samples)
    with torch.no_grad():
        for _, r in sub.iterrows():
            t0 = time.perf_counter()
            try:
                im = Image.open(os.path.join(DATASET, r.image_path)).convert("RGB")
            except Exception:
                im = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            x = TF(im).unsqueeze(0)
            fi = img_enc(x)
            t = re.sub(r"http\S+|www\S+|https\S+", "", str(r.tweet_text), flags=re.IGNORECASE)
            t = re.sub(r"\s+", " ", t).strip()
            e = tok(t if t else " ", truncation=True, padding="max_length",
                    max_length=64, return_tensors="pt")
            ft = bert(input_ids=e["input_ids"], attention_mask=e["attention_mask"]) \
                .last_hidden_state[:, 0]
            z, _ = head(fi, ft)
            _ = z.argmax(1)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def final_eval(ckpt_name, cache, biases=None):
    """Single frozen evaluation of the selected final model on TEST."""
    ckpt_dir = os.path.join(EXP_DIR, ckpt_name)
    state = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location="cpu")

    rmeta = json.load(open(os.path.join(ckpt_dir, "result.json")))
    head = rmeta["head"]
    model = GatedFusionWide(dropout=rmeta.get("dropout", 0.2))
    if head == "coral":
        model.enable_coral()
    model.load_state_dict(state)
    model.eval()

    im, tx, y = cache["img_feats"], cache["txt_feats"], cache["labels"]
    iv, it = cache["idx_val"], cache["idx_test"]

    def run(idx, bias_vec):
        loader = DataLoader(FeatDS(im[idx], tx[idx], y[idx]), batch_size=64)
        ys, ps, prob = [], [], []
        with torch.no_grad():
            for a, b, yy in loader:
                z, _ = model(a, b)
                if head == "coral":
                    pr = coral_predict(z)
                    pb = torch.sigmoid(z).numpy()
                else:
                    pr = z.argmax(1)
                    pb = F.softmax(z, 1).numpy()
                ys += list(yy.numpy()); ps += list(pr.numpy()); prob.append(pb)
        prob = np.concatenate(prob)
        if bias_vec is not None:
            ps = (prob + bias_vec).argmax(1)
        return np.array(ys), np.array(ps)

    yt, yp = run(it, None)
    m = metrics_dict(yt, yp)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    per_class = {}
    for i, ln in enumerate(LABEL_NAMES):
        p, r, f, _ = precision_recall_fscore_support(yt == i, yp == i, average="binary", zero_division=0)
        per_class[ln] = {"precision": float(p), "recall": float(r), "f1": float(f),
                         "support": int((yt == i).sum())}
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)

    # fusion-head-only latency (batch=1) over test set
    te_loader = DataLoader(FeatDS(im[it], tx[it], y[it]), batch_size=1)
    lat = []
    with torch.no_grad():
        for a, b, _ in te_loader:
            t0 = time.perf_counter()
            z, _ = model(a, b); _ = z.argmax(1)
            lat.append((time.perf_counter() - t0) * 1000)

    out = {
        "checkpoint": ckpt_name, "config": rmeta,
        "test_metrics": {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
            "precision_weighted": float(pw), "recall_weighted": float(rw),
            "f1_weighted": float(fw)},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "fusion_head_latency_ms_mean": float(np.mean(lat)),
        "fusion_head_latency_ms_std": float(np.std(lat)),
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dapt")
    ap.add_argument("--recipe", default="", help="loss,sampling,noise e.g. wce,bal,0.005")
    ap.add_argument("--final_ckpt", default="")
    a = ap.parse_args()

    cache = torch.load(os.path.join(EXP_DIR, f"features_{a.arm}.pt"), map_location="cpu",
                       weights_only=False)

    if a.recipe:
        ln, sp, nz = a.recipe.split(",")
        nz = float(nz)
        outs = {}
        for kind in ["img", "txt", "concat", "v1", "v2"]:
            outs[kind] = train_abl(f"{kind}", kind, cache, ln, sp, nz)
        json.dump(outs, open(os.path.join(EXP_DIR, "ablation_results.json"), "w"),
                  indent=2, default=float)

    if a.final_ckpt:
        res = final_eval(a.final_ckpt, cache)
        opt_split = pd.read_csv(os.path.join(ROOT, "research", "optimization", "cleaned_dataset.csv"))
        te_df = opt_split[opt_split.split == "test"].reset_index(drop=True)
        mp, sp_ = full_pipeline_latency(os.path.join(EXP_DIR, a.final_ckpt), te_df)
        res["full_pipeline_latency_ms_mean"] = mp
        res["full_pipeline_latency_ms_std"] = sp_
        json.dump(res, open(os.path.join(EXP_DIR, "FINAL_TEST_RESULTS.json"), "w"),
                  indent=2, default=float)
        print(json.dumps(res["test_metrics"], indent=2))
        print("CM:", res["confusion_matrix"])
