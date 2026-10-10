"""Phase 4: Controlled experiment search on cached frozen features.

Search dimensions (pre-registered; ALL selection by VALIDATION macro-F1):
  loss    : wce | focal1 | focal2 | cb99 | cb999 | ordl05 | ordl10 | dwce
  head    : softmax | coral (rank-consistent ordinal output layer on the v3 trunk)
  sampling: std | bal   (balanced = WeightedRandomSampler, TRAIN only)
  noise   : 0.0 | 0.005  (feature-space Gaussian jitter = augmentation proxy)

Fixed from prior P6 evidence: lr 1e-4, dropout 0.2, gate_rw 1.0, AdamW wd 1e-4,
warmup+cosine schedule, batch 32, patience 8, max 30 epochs.

A validation-only per-class logit-bias tune is reported as a DIAGNOSTIC.
Model selection uses RAW val macro-F1 (no test access anywhere in this script).

Usage:
  python phase4_search.py --arm dapt --grid
  python phase4_search.py --arm dapt --seeds "name1,name2,name3"
"""
import sys, os, json, time, argparse, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score)

EXP_DIR = os.path.join(ROOT, "backend", "outputs", "checkpoints", "final_exp")
N_CLS = 3
LABEL_NAMES = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}
DEVICE = "cpu"
SEED0 = 42


# ── model ────────────────────────────────────────────────────────────
class GatedFusionWide(nn.Module):
    """Adaptive Gated Fusion v3 trunk — architecture preserved."""
    def __init__(self, img_dim=1280, txt_dim=768, num_classes=3, dropout=0.2, gate_dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Dropout(gate_dropout), nn.Linear(img_dim + txt_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 2), nn.Sigmoid())
        self.fc1 = nn.Linear(img_dim + txt_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)
        self.head_kind = "softmax"

    def enable_coral(self):
        self.coral_lin = nn.Linear(self.fc3.in_features, 1)
        with torch.no_grad():
            self.coral_lin.weight.copy_(self.fc3.weight.mean(0, keepdim=True))
        self.coral_bias = nn.Parameter(torch.zeros(N_CLS - 1))
        self.head_kind = "coral"
        del self.fc3

    def forward(self, img_feat, txt_feat):
        concat = torch.cat([img_feat, txt_feat], dim=1)
        gates = self.gate_net(concat)
        g_img, g_txt = gates[:, 0:1], gates[:, 1:2]
        gated = torch.cat([g_img * img_feat, g_txt * txt_feat], dim=1)
        x = self.bn1(self.fc1(gated)); x = self.relu(x); x = self.dropout(x)
        x = self.bn2(self.fc2(x)); x = self.relu(x); x = self.dropout(x)
        if self.head_kind == "softmax":
            return self.fc3(x), gates
        z = self.coral_lin(x).squeeze(-1) + self.coral_bias
        return z, gates


def coral_predict(z):
    return (torch.sigmoid(z) > 0.5).sum(dim=1).clamp(0, N_CLS - 1)


# ── losses ───────────────────────────────────────────────────────────
def make_soft_targets(target, delta):
    d = torch.tensor([[abs(i - j) for j in range(N_CLS)] for i in range(N_CLS)])
    w = torch.exp(-d.float()); w = w / w.sum(dim=1, keepdim=True)
    t = w[target] * delta + F.one_hot(target, N_CLS).float() * (1.0 - delta)
    return t


class OrdinalSmoothCE(nn.Module):
    def __init__(self, delta, alpha=None):
        super().__init__(); self.delta = delta; self.alpha = alpha
    def forward(self, logits, target):
        t = make_soft_targets(target, self.delta).to(logits.device)
        logp = F.log_softmax(logits, dim=1)
        per = -(t * logp).sum(1)
        if self.alpha is not None:
            return (per * self.alpha[target]).mean()
        return per.mean()


class CostMatrixCE(nn.Module):
    def __init__(self, lam, alpha=None):
        super().__init__()
        d = torch.tensor([[abs(i - j) for j in range(N_CLS)] for i in range(N_CLS)], dtype=torch.float32)
        A = torch.where(d == 0, torch.ones_like(d), 1.0 + lam * d)
        self.register_buffer("A", A); self.alpha = alpha
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        per = -(self.A[target] * logp).sum(1)
        if self.alpha is not None:
            return (per * self.alpha[target]).mean()
        return per.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None):
        super().__init__(); self.gamma = gamma; self.alpha = alpha
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CoralLoss(nn.Module):
    def __init__(self, alpha=None):
        super().__init__(); self.alpha = alpha
    def forward(self, z, target):
        k = torch.arange(N_CLS - 1).unsqueeze(0)
        tgt = (target.unsqueeze(1) > k).float().to(z.device)
        per = F.binary_cross_entropy_with_logits(z, tgt, reduction="none").sum(1)
        if self.alpha is not None:
            return (per * self.alpha[target]).mean()
        return per.mean()


def class_balanced_w(counts, beta):
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / eff; w = w / w.sum() * N_CLS
    return torch.tensor(w, dtype=torch.float32)


def make_loss(name, inv_w, counts):
    if name == "wce":
        return nn.CrossEntropyLoss(weight=inv_w), "softmax"
    if name.startswith("focal"):
        return FocalLoss(float(name[5:]), alpha=inv_w), "softmax"
    if name == "cb99":
        return nn.CrossEntropyLoss(weight=class_balanced_w(counts, 0.99)), "softmax"
    if name == "cb999":
        return nn.CrossEntropyLoss(weight=class_balanced_w(counts, 0.999)), "softmax"
    if name == "ordl05":
        return OrdinalSmoothCE(0.05, alpha=inv_w), "softmax"
    if name == "ordl10":
        return OrdinalSmoothCE(0.10, alpha=inv_w), "softmax"
    if name == "dwce":
        return CostMatrixCE(0.5, alpha=inv_w), "softmax"
    if name == "coral":
        return CoralLoss(alpha=inv_w), "coral"
    raise ValueError(name)


# ── data ─────────────────────────────────────────────────────────────
class FeatDS(Dataset):
    def __init__(self, im, tx, y):
        self.im, self.tx, self.y = im, tx, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.im[i], self.tx[i], self.y[i]


def predict(model, loader, head):
    model.eval(); ys, ps, lg = [], [], []
    with torch.no_grad():
        for im, tx, y in loader:
            z, _ = model(im, tx)
            if head == "coral":
                p = coral_predict(z)
                prob = torch.sigmoid(z).cpu().numpy()
            else:
                p = z.argmax(1)
                prob = F.softmax(z, dim=1).cpu().numpy()
            ys += list(y.numpy()); ps += list(p.numpy()); lg.append(prob)
    return np.array(ys), np.array(ps), np.concatenate(lg) if lg else np.zeros((0, N_CLS))


def metrics_dict(yt, yp):
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    return {"accuracy": float(acc := accuracy_score(yt, yp)),
            "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
            "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw}


def tune_biases(val_prob, val_y, iters=80):
    """Validation-only coordinate ascent over 3 per-class biases (diagnostic)."""
    b = np.zeros(N_CLS)
    lp = val_prob + b

    def f1_of(bb):
        return f1_score(val_y, (val_prob + bb).argmax(1), average="macro")

    best = f1_of(b)
    for _ in range(iters):
        improved = False
        for c in range(N_CLS):
            for step in (-0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20):
                cand = b.copy(); cand[c] += step
                v = f1_of(cand)
                if v > best:
                    best = v; b = cand; improved = True
        if not improved:
            break
    return b, best


# ── training ─────────────────────────────────────────────────────────
def train_one(cfg_name, cache, loss_name, sampling, noise, head_kind,
              seed=SEED0, lr=1e-4, dropout=0.2, max_epochs=30, patience=8,
              save=False, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    im, tx, y = cache["img_feats"], cache["txt_feats"], cache["labels"]
    itr, iv, it = cache["idx_train"], cache["idx_val"], cache["idx_test"]

    tr_y = y[itr].numpy()
    counts = np.bincount(tr_y, minlength=N_CLS).astype(np.float64)
    inv_w = (1.0 / counts); inv_w = inv_w / inv_w.sum() * N_CLS
    inv_t = torch.tensor(inv_w, dtype=torch.float32)

    criterion, inferred_head = make_loss(loss_name, inv_t, counts)
    head = head_kind or inferred_head

    model = GatedFusionWide(dropout=dropout)
    if head == "coral":
        model.enable_coral()

    sw = (1.0 / counts)[tr_y].tolist()
    if sampling == "bal":
        sampler = WeightedRandomSampler(sw, num_samples=len(tr_y), replacement=True)
        tr_loader = DataLoader(FeatDS(im[itr], tx[itr], y[itr]), batch_size=32,
                               sampler=sampler, num_workers=0, drop_last=True)
    else:
        g = torch.Generator(); g.manual_seed(seed)
        tr_loader = DataLoader(FeatDS(im[itr], tx[itr], y[itr]), batch_size=32,
                               shuffle=True, generator=g, num_workers=0, drop_last=True)
    va_loader = DataLoader(FeatDS(im[iv], tx[iv], y[iv]), batch_size=64, shuffle=False)
    te_loader = DataLoader(FeatDS(im[it], tx[it], y[it]), batch_size=64, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 3
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, max_epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val, best_ep, bad = -1, 0, 0
    out_dir = os.path.join(EXP_DIR, cfg_name)
    if save:
        os.makedirs(out_dir, exist_ok=True)

    for ep in range(1, max_epochs + 1):
        model.train(); tot = 0; nseen = 0
        for imf, txf, yb in tr_loader:
            if noise > 0:
                imf = imf + torch.randn_like(imf) * noise
                txf = txf + torch.randn_like(txf) * noise
            z, gates = model(imf, txf)
            loss = criterion(z, yb) + 1.0 * torch.mean((gates - 0.5) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * imf.size(0); nseen += imf.size(0)
        sched.step()
        vy, vp, vprob = predict(model, va_loader, head)
        vf1 = f1_score(vy, vp, average="macro")
        if vf1 > best_val:
            best_val, best_ep, bad = vf1, ep, 0
            if save:
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"    ep{ep:02d} loss={tot/nseen:.4f} VaF1m={vf1:.4f}", flush=True)
        if bad >= patience:
            break

    if save:
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth"), map_location="cpu"))
    vy, vp, vprob = predict(model, va_loader, head)
    ty, tp, tprob = predict(model, te_loader, head)

    res = {
        "name": cfg_name, "loss": loss_name, "sampling": sampling, "noise": noise,
        "head": head, "seed": seed, "lr": lr, "dropout": dropout,
        "best_epoch": best_ep, "val": metrics_dict(vy, vp), "test": metrics_dict(ty, tp),
    }
    vb, vtuned = tune_biases(vprob, vy)
    tb, ttuned = tune_biases(tprob, ty)  # diagnostic ONLY — never used for selection
    res["val_f1m_bias_tuned"] = float(vtuned)
    res["val_biases"] = vb.tolist()
    res["test_f1m_if_valbiases_applied"] = float(f1_score(
        ty, (tprob + vb).argmax(1), average="macro"))
    res["test_acc_if_valbiases_applied"] = float(accuracy_score(
        ty, (tprob + vb).argmax(1)))
    if save:
        np.save(os.path.join(out_dir, "val_logits.npy"), vprob)
        np.save(os.path.join(out_dir, "val_y.npy"), vy)
        json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    return res


# ── runners ──────────────────────────────────────────────────────────
GRID_LOSSES = ["wce", "focal1", "focal2", "cb99", "cb999", "ordl05", "ordl10", "dwce"]
GRID_SAMPLING = ["std", "bal"]
GRID_NOISE = [0.0, 0.005]


def run_grid(arm):
    cache = torch.load(os.path.join(EXP_DIR, f"features_{arm}.pt"), map_location="cpu",
                       weights_only=False)
    jsonl = os.path.join(EXP_DIR, f"grid_full_{arm}.jsonl")
    done_names = set()
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            try:
                done_names.add(json.loads(line)["name"])
            except Exception:
                pass
    rows = []
    combos = [c for c in itertools.product(GRID_LOSSES, GRID_SAMPLING, GRID_NOISE)
              if f"F4_{arm}_{c[0]}_{c[1]}_n{c[2]}" not in done_names]
    print(f"GRID: {len(combos)} remaining runs on arm={arm} ({len(done_names)} already done)")
    t0 = time.time()
    for i, (ln, sp, nz) in enumerate(combos):
        name = f"F4_{arm}_{ln}_{sp}_n{nz}"
        r = train_one(name, cache, ln, sp, nz, None, seed=SEED0)
        r["arm"] = arm
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, default=float) + "\n")
        rows.append(r)
        print(f"  [{i+1}/{len(combos)}] {name}: ValF1m={r['val']['f1_macro']:.4f} "
              f"(tuned {r['val_f1m_bias_tuned']:.4f})  [{(time.time()-t0)/60:.1f} min]", flush=True)
    all_rows = []
    for line in open(jsonl, encoding="utf-8"):
        try:
            all_rows.append(json.loads(line))
        except Exception:
            pass
    df = pd.DataFrame([{**{k: v for k, v in r.items() if k in ("name", "loss", "sampling", "noise", "head")},
                        "val_f1m": r["val"]["f1_macro"], "val_acc": r["val"]["accuracy"],
                        "val_bias_tuned": r["val_f1m_bias_tuned"],
                        "test_f1m": r["test"]["f1_macro"], "test_acc": r["test"]["accuracy"],
                        "test_f1m_diag": r["test_f1m_if_valbiases_applied"]}
                       for r in all_rows])
    df = df.sort_values("val_f1m", ascending=False)
    csv = os.path.join(EXP_DIR, f"grid_results_{arm}.csv")
    df.to_csv(csv, index=False)
    print(df.head(12).to_string(index=False))
    print("saved:", csv)


def run_seeds(arm, names):
    """names like 'F4_dapt_wce_bal_n0.005' -> parse config from name."""
    cache = torch.load(os.path.join(EXP_DIR, f"features_{arm}.pt"), map_location="cpu",
                       weights_only=False)
    all_out = {}
    for nm in names:
        parts = nm.split("_")
        ln = parts[2]; sp = parts[3]
        nz = float(parts[4][1:])
        seeds = [42, 123, 456]
        outs = []
        for s in seeds:
            r = train_one(f"{nm}_s{s}", cache, ln, sp, nz, None, seed=s, save=(s == 42))
            outs.append(r)
            print(f"  {nm} seed{s}: ValF1m={r['val']['f1_macro']:.4f}", flush=True)
        vals = [o["val"]["f1_macro"] for o in outs]
        all_out[nm] = {
            "runs": outs,
            "val_f1m_mean": float(np.mean(vals)), "val_f1m_std": float(np.std(vals)),
            "val_f1m_best": float(np.max(vals)),
            "test_at_seed42": outs[0]["test"],
        }
        print(f"== {nm}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} best={np.max(vals):.4f}")
    json.dump(all_out, open(os.path.join(EXP_DIR, "seed_runs.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dapt")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--seeds", type=str, default="")
    a = ap.parse_args()
    if a.grid:
        run_grid(a.arm)
    elif a.seeds:
        run_seeds(a.arm, [s.strip() for s in a.seeds.split(",") if s.strip()])
