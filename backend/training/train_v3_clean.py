"""Retrain Gated Fusion v3 on the cleaned dataset.

Reproduces the published best v3 config `wide_wce_g1.0_lr1e-04` from
train_v3_batch2.py exactly:
  - architecture: GatedFusionWide (1024 hidden, BN, dropout 0.3, gate_dropout 0.1)
  - loss: WeightedCE with inverse-frequency class weights (normalized to sum=3)
  - gate regularization: target 0.5, weight 1.0
  - optimizer: AdamW, lr=1e-4, weight_decay=1e-4
  - scheduler: warmup_cosine (5-epoch warmup, cosine to 1e-6)
  - sampler: WeightedRandomSampler (inverse), replacement=True
  - batch: train 32, val/test 64; feature noise 0.0
  - patience 10, max epochs 40; early stop on val macro-F1
  - seed: RANDOM_SEED (42)
Data: backend/outputs/checkpoints/v3_clean/feature_cache.pt (cleaned population,
leak-free split). Frozen encoders unchanged (EfficientNet-B0 + DistilBERT).

Outputs (backend/outputs/checkpoints/v3_clean/):
  cleaned_v3_wide_wce_g1.0_lr1e-04_best.pth
  cleaned_v3_wide_wce_g1.0_lr1e-04_summary.json
  cleaned_v3_wide_wce_g1.0_lr1e-04_cm.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, f1_score,
)

from backend.training.config import (
    LABEL_NAMES, NUM_CLASSES, CHECKPOINTS_DIR, RANDOM_SEED,
)
from backend.models.fusion_models import GatedFusionWide, count_parameters
from backend.config.shared import DEVICE

SEVERITY_LABELS = ["Little or No Damage", "Mild Damage", "Severe Damage"]
DEV = torch.device(DEVICE)
OUT_DIR = CHECKPOINTS_DIR / "v3_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_NAME = "cleaned_v3_wide_wce_g1.0_lr1e-04"
CACHE_PATH = OUT_DIR / "feature_cache.pt"

LR = 1e-4
WEIGHT_DECAY = 1e-4
GATE_RW = 1.0
GATE_TARGET = 0.5
PATIENCE = 10
MAX_EPOCHS = 40
TRAIN_BATCH = 32
EVAL_BATCH = 64
FEATURE_NOISE = 0.0


class WeightedCE(nn.Module):
    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.alpha)


def gate_reg(gates, target=GATE_TARGET, weight=GATE_RW):
    return weight * torch.mean((gates - target) ** 2)


class FusionDS(Dataset):
    def __init__(self, img_f, txt_f, labels):
        self.img_f, self.txt_f, self.labels = img_f, txt_f, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.img_f[idx], self.txt_f[idx], self.labels[idx]


def collate(b):
    return (
        torch.stack([x[0] for x in b]),
        torch.stack([x[1] for x in b]),
        torch.stack([x[2] for x in b]),
    )


def eval_full(model, loader):
    model.eval()
    preds_l, labels_l, gates_l = [], [], []
    with torch.no_grad():
        for img_f, txt_f, labels in loader:
            img_f, txt_f, labels = img_f.to(DEV), txt_f.to(DEV), labels.to(DEV)
            logits, gates = model(img_f, txt_f)
            preds_l.extend(logits.argmax(1).cpu().numpy())
            labels_l.extend(labels.cpu().numpy())
            gates_l.append(gates.cpu())
    yt = np.array(labels_l)
    yp = np.array(preds_l)
    gc = torch.cat(gates_l)
    acc = accuracy_score(yt, yp)
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    cr = classification_report(yt, yp, target_names=SEVERITY_LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": acc, "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "img_gate": gc[:, 0].mean().item(), "txt_gate": gc[:, 1].mean().item(),
    }, cr, cm, yt, yp


class ES:
    def __init__(self, patience=10):
        self.patience = patience
        self.c = 0
        self.best = None
        self.stop = False

    def __call__(self, score):
        if self.best is None or score > self.best:
            self.best = score
            self.c = 0
            return False
        self.c += 1
        if self.c >= self.patience:
            self.stop = True
        return True


def main():
    t0 = time.time()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print(f"  Retrain: {RUN_NAME}")
    print(f"  Architecture: GatedFusionWide | Loss: WCE(inverse) | GateReg: {GATE_RW} | "
          f"LR: {LR} | WD: {WEIGHT_DECAY} | Patience: {PATIENCE} | MaxEp: {MAX_EPOCHS}")
    print("=" * 70)

    cache = torch.load(CACHE_PATH, weights_only=False)
    img_all = cache["img_feats"]
    txt_all = cache["txt_feats"]
    labels_all = cache["labels"]
    train_idx = cache["train_idx"]
    val_idx = cache["val_idx"]
    test_idx = cache["test_idx"]
    print(f"  Population: {labels_all.numel()} | "
          f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_l = labels_all[train_idx]
    class_counts = torch.bincount(train_l, minlength=NUM_CLASSES).float()
    print(f"  Train class counts: {class_counts.long().tolist()}")

    weights_inv = 1.0 / class_counts
    weights_inv = weights_inv / weights_inv.sum() * NUM_CLASSES
    print(f"  WCE weights (inverse, normalized): {[round(w, 4) for w in weights_inv.tolist()]}")

    sw_inv = (1.0 / class_counts)[train_l].tolist()
    sampler_inv = WeightedRandomSampler(sw_inv, num_samples=len(train_l), replacement=True)

    train_ds = FusionDS(img_all[train_idx], txt_all[train_idx], labels_all[train_idx])
    val_ds = FusionDS(img_all[val_idx], txt_all[val_idx], labels_all[val_idx])
    test_ds = FusionDS(img_all[test_idx], txt_all[test_idx], labels_all[test_idx])

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, sampler=sampler_inv,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH, shuffle=False,
                            collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False,
                             collate_fn=collate, num_workers=0)

    model = GatedFusionWide(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES,
                            dropout=0.3, gate_dropout=0.1).to(DEV)
    tr, tt = count_parameters(model)
    print(f"  Params: {tr:,} trainable / {tt:,} total")

    criterion = WeightedCE(alpha=weights_inv.to(DEV))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup = min(5, MAX_EPOCHS // 5)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, MAX_EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    es = ES(patience=PATIENCE)
    hist = {"train_loss": [], "val_f1": [], "img_gate": [], "txt_gate": []}
    best_f1, best_ep = 0.0, 0

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        tp, tl = [], []

        for img_f, txt_f, labels in train_loader:
            img_f, txt_f, labels = img_f.to(DEV), txt_f.to(DEV), labels.to(DEV)
            logits, gates = model(img_f, txt_f)
            loss = criterion(logits, labels) + gate_reg(gates)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * img_f.size(0)
            tp.extend(logits.argmax(1).cpu().numpy())
            tl.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_ds)
        train_f1 = f1_score(tl, tp, average="macro")

        vm, vcr, vcm, _, _ = eval_full(model, val_loader)
        scheduler.step()

        hist["train_loss"].append(train_loss)
        hist["val_f1"].append(vm["f1_macro"])
        hist["img_gate"].append(vm["img_gate"])
        hist["txt_gate"].append(vm["txt_gate"])

        imp = ""
        if vm["f1_macro"] > best_f1:
            best_f1 = vm["f1_macro"]
            best_ep = ep
            torch.save(model.state_dict(), OUT_DIR / f"{RUN_NAME}_best.pth")
            imp = " *** BEST ***"

        if ep % 5 == 0 or imp or ep == 1:
            print(f"  Ep {ep:2d}  TrL={train_loss:.4f} TrF1={train_f1:.4f}  "
                  f"VaF1={vm['f1_macro']:.4f} VaAcc={vm['accuracy']:.4f}  "
                  f"G[I={vm['img_gate']:.3f},T={vm['txt_gate']:.3f}]{imp}")

        if es(vm["f1_macro"]):
            print(f"  Early stop at ep {ep} (best ep {best_ep})")
            break

    model.load_state_dict(torch.load(OUT_DIR / f"{RUN_NAME}_best.pth", map_location=DEV))
    tm, tcr, tcm, yt, yp = eval_full(model, test_loader)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(tcm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=ax)
    ax.set_title(f"Cleaned v3 — {RUN_NAME}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(OUT_DIR / f"{RUN_NAME}_cm.png", dpi=150); plt.close(fig)

    summary = {"name": RUN_NAME, "best_epoch": best_ep}
    summary.update({k: round(float(v), 4) for k, v in tm.items()})
    for lbl in range(3):
        dn = SEVERITY_LABELS[lbl]
        sn = LABEL_NAMES[lbl]
        summary[f"precision_{sn}"] = round(tcr[dn]["precision"], 4)
        summary[f"recall_{sn}"] = round(tcr[dn]["recall"], 4)
        summary[f"f1_{sn}"] = round(tcr[dn]["f1-score"], 4)

    summary["config"] = {
        "model": "GatedFusionWide",
        "loss": "WeightedCE(inverse class frequencies)",
        "class_weights": weights_inv.tolist(),
        "gate_reg_weight": GATE_RW, "gate_target": GATE_TARGET,
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "scheduler": "warmup_cosine", "warmup_epochs": warmup,
        "sampler": "WeightedRandomSampler(inverse, replacement=True)",
        "train_batch": TRAIN_BATCH, "eval_batch": EVAL_BATCH,
        "feature_noise": FEATURE_NOISE, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "seed": RANDOM_SEED,
        "population": "cleaned CrisisMMD subset (n=3105)",
        "split": "leak-free stratified 70/15/15 (Phase 4)",
        "train_val_test": [int(len(train_idx)), int(len(val_idx)), int(len(test_idx))],
        "hardware": "CPU", "wall_seconds": round(time.time() - t0, 1),
    }

    with open(OUT_DIR / f"{RUN_NAME}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # persist preds for phase 6/7
    np.savez(OUT_DIR / f"{RUN_NAME}_preds.npz", yt=yt, yp=yp)

    print(f"\n  TEST: Acc={tm['accuracy']:.4f} F1m={tm['f1_macro']:.4f} F1w={tm['f1_weighted']:.4f} "
          f"Gates[I={tm['img_gate']:.4f},T={tm['txt_gate']:.4f}]")
    for lbl in range(3):
        dn = SEVERITY_LABELS[lbl]
        print(f"    {dn:25s}  P={tcr[dn]['precision']:.4f}  R={tcr[dn]['recall']:.4f}  F1={tcr[dn]['f1-score']:.4f}")
    print(f"\n  Wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
