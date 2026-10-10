"""CrisisMind v3 Batch 2 — Targeted improvements.

Based on batch 1 findings (16 runs, best=0.5020, all WCE+inverse):
- No config beat v1 (0.5375) or v2 (0.5256)
- Longer training (60 ep) may overfit with frozen features
- Need fundamentally different approaches

New strategies:
1. Label smoothing (reduces overconfidence on majority class)
2. Larger fusion head (more capacity)
3. Sample reweighting (harder examples get more weight)
4. Reduced max epochs (30-40) + lower patience
5. Different loss functions with proper configs
6. Feature dropout / noise injection

Usage: python train_v3_batch2.py [batch_num]
  batch 0: label smoothing + focal combos
  batch 1: larger head + new architectures  
  batch 2: class-balanced loss + manual weights
  batch 3: mixup + advanced augmentation
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json, time, shutil
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
    LABEL_NAMES, NUM_CLASSES, FUSION_DROPOUT, CHECKPOINTS_DIR, RANDOM_SEED,
)
from backend.models.fusion_models import GatedFusion, count_parameters
from backend.config.shared import DEVICE

SEVERITY_LABELS = ["Little or No Damage", "Mild Damage", "Severe Damage"]
DEV = torch.device(DEVICE)
V3_DIR = CHECKPOINTS_DIR / "v3"
V2_DIR = CHECKPOINTS_DIR / "v2"
RESULTS_DIR = Path("testing")

V1_BASELINE = {
    "accuracy": 0.6352, "f1_macro": 0.5375, "f1_weighted": 0.6415,
    "recall_little_or_no_damage": 0.4366, "recall_mild_damage": 0.4444, "recall_severe_damage": 0.7500,
    "img_gate": 0.173, "txt_gate": 0.014,
}
V2_BASELINE = {
    "accuracy": 0.5766, "f1_macro": 0.5256, "f1_weighted": 0.6059,
    "recall_little_or_no_damage": 0.5634, "recall_mild_damage": 0.4921, "recall_severe_damage": 0.6235,
    "img_gate": 0.644, "txt_gate": 0.431,
}


# ═══════════════════════════════════════════════════════════════════
# Custom GatedFusion variants
# ═══════════════════════════════════════════════════════════════════
class GatedFusionWide(nn.Module):
    """Larger fusion head (1024 hidden) + dropout on gate input."""
    def __init__(self, img_dim=1280, txt_dim=768, num_classes=3, dropout=0.3, gate_dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Dropout(gate_dropout),
            nn.Linear(img_dim + txt_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(img_dim + txt_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, img_feat, txt_feat):
        concat = torch.cat([img_feat, txt_feat], dim=1)
        gates = self.gate_net(concat)
        g_img, g_txt = gates[:, 0:1], gates[:, 1:2]
        gated = torch.cat([g_img * img_feat, g_txt * txt_feat], dim=1)
        x = self.bn1(self.fc1(gated))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x, gates


class GatedFusionDeep(nn.Module):
    """3-layer fusion with residual-like skip."""
    def __init__(self, img_dim=1280, txt_dim=768, num_classes=3, dropout=0.3):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + txt_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(img_dim + txt_dim, 768)
        self.bn1 = nn.BatchNorm1d(768)
        self.fc2 = nn.Linear(768, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feat, txt_feat):
        concat = torch.cat([img_feat, txt_feat], dim=1)
        gates = self.gate_net(concat)
        g_img, g_txt = gates[:, 0:1], gates[:, 1:2]
        x = self.bn1(self.fc1(torch.cat([g_img * img_feat, g_txt * txt_feat], dim=1)))
        x = self.relu(x); x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x); x = self.dropout(x)
        x = self.bn3(self.fc3(x))
        x = self.relu(x); x = self.dropout(x)
        x = self.fc4(x)
        return x, gates


# ═══════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════
class WeightedCE(torch.nn.Module):
    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.alpha)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class LabelSmoothingCE(torch.nn.Module):
    def __init__(self, alpha=None, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.smoothing = smoothing
    def forward(self, logits, targets):
        n = logits.size(0)
        c = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        # Smooth targets
        smooth = torch.full_like(log_probs, self.smoothing / c)
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / c)
        if self.alpha is not None:
            loss = -(self.alpha.unsqueeze(0) * smooth * log_probs).sum(dim=1)
        else:
            loss = -(smooth * log_probs).sum(dim=1)
        return loss.mean()


class ClassBalancedLoss(torch.nn.Module):
    def __init__(self, samples_per_class, beta=0.999, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        counts = torch.tensor(samples_per_class, dtype=torch.float)
        effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / torch.clamp(effective, min=1e-8)
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("alpha", weights)
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ═══════════════════════════════════════════════════════════════════
# Gate reg + feature noise
# ═══════════════════════════════════════════════════════════════════
def gate_reg(gates, target=0.5, weight=0.01):
    return weight * torch.mean((gates - target) ** 2)


def add_feature_noise(img_f, txt_f, noise_std=0.01):
    if noise_std > 0:
        img_f = img_f + torch.randn_like(img_f) * noise_std
        txt_f = txt_f + torch.randn_like(txt_f) * noise_std
    return img_f, txt_f


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════
# Eval
# ═══════════════════════════════════════════════════════════════════
def eval_full(model, loader, feature_noise=0.0):
    model.eval()
    preds_l, labels_l, gates_l = [], [], []
    with torch.no_grad():
        for img_f, txt_f, labels in loader:
            img_f, txt_f, labels = img_f.to(DEV), txt_f.to(DEV), labels.to(DEV)
            if feature_noise > 0:
                img_f, txt_f = add_feature_noise(img_f, txt_f, feature_noise)
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


# ═══════════════════════════════════════════════════════════════════
# Single training run
# ═══════════════════════════════════════════════════════════════════
def train_run(name, train_loader, val_loader, test_loader,
              model_cls, model_kwargs, criterion, gate_rw, lr,
              weight_decay, patience, max_epochs, scheduler_type,
              feature_noise=0.0):
    print(f"\n{'='*65}")
    print(f"  RUN: {name}")
    print(f"{'='*65}")

    model = model_cls(**model_kwargs).to(DEV)
    t, tt = count_parameters(model)
    print(f"  Params: {t:,} trainable / {tt:,} total")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    elif scheduler_type == "warmup_cosine":
        warmup = min(5, max_epochs // 5)
        def lr_lambda(ep):
            if ep < warmup:
                return (ep + 1) / warmup
            progress = (ep - warmup) / max(1, max_epochs - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    es = ES(patience=patience)
    hist = {"train_loss": [], "val_f1": [], "img_gate": [], "txt_gate": []}
    best_f1 = 0.0
    best_ep = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        total_loss = 0
        tp, tl = [], []

        for img_f, txt_f, labels in train_loader:
            img_f, txt_f, labels = img_f.to(DEV), txt_f.to(DEV), labels.to(DEV)
            if feature_noise > 0:
                img_f, txt_f = add_feature_noise(img_f, txt_f, feature_noise)
            logits, gates = model(img_f, txt_f)
            loss = criterion(logits, labels) + gate_reg(gates, 0.5, gate_rw)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * img_f.size(0)
            tp.extend(logits.argmax(1).cpu().numpy())
            tl.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader.dataset)
        train_f1 = f1_score(tl, tp, average="macro")

        vm, vcr, vcm, _, _ = eval_full(model, val_loader, feature_noise)
        scheduler.step()

        hist["train_loss"].append(train_loss)
        hist["val_f1"].append(vm["f1_macro"])
        hist["img_gate"].append(vm["img_gate"])
        hist["txt_gate"].append(vm["txt_gate"])

        imp = ""
        if vm["f1_macro"] > best_f1:
            best_f1 = vm["f1_macro"]
            best_ep = ep
            torch.save(model.state_dict(), V3_DIR / f"{name}_best.pth")
            imp = " *** BEST ***"

        if ep % 5 == 0 or imp or ep == 1:
            print(f"  Ep {ep:2d}  TrL={train_loss:.4f} TrF1={train_f1:.4f}  "
                  f"VaF1={vm['f1_macro']:.4f} VaAcc={vm['accuracy']:.4f}  "
                  f"G[I={vm['img_gate']:.3f},T={vm['txt_gate']:.3f}]{imp}")

        if es(vm["f1_macro"]):
            print(f"  Early stop at ep {ep} (best ep {best_ep})")
            break

    model.load_state_dict(torch.load(V3_DIR / f"{name}_best.pth", map_location=DEV))
    tm, tcr, tcm, yt, yp = eval_full(model, test_loader, feature_noise)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(tcm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=ax)
    ax.set_title(f"CrisisMind v3 — {name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(V3_DIR / f"{name}_cm.png", dpi=150); plt.close(fig)

    summary = {"name": name, "best_epoch": best_ep}
    summary.update({k: round(float(v), 4) for k, v in tm.items()})
    for lbl in range(3):
        dn = SEVERITY_LABELS[lbl]
        sn = LABEL_NAMES[lbl]
        summary[f"precision_{sn}"] = round(tcr[dn]["precision"], 4)
        summary[f"recall_{sn}"] = round(tcr[dn]["recall"], 4)
        summary[f"f1_{sn}"] = round(tcr[dn]["f1-score"], 4)

    with open(V3_DIR / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  TEST: Acc={tm['accuracy']:.4f} F1m={tm['f1_macro']:.4f} F1w={tm['f1_weighted']:.4f} "
          f"Gates[I={tm['img_gate']:.4f},T={tm['txt_gate']:.4f}]")
    for lbl in range(3):
        dn = SEVERITY_LABELS[lbl]
        print(f"    {dn:25s}  P={tcr[dn]['precision']:.4f}  R={tcr[dn]['recall']:.4f}  F1={tcr[dn]['f1-score']:.4f}")
    return summary


# ═══════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════
def load_data():
    cache_path = V2_DIR / "feature_cache.pt"
    if not cache_path.exists():
        cache_path = V3_DIR / "feature_cache.pt"
    if not cache_path.exists():
        print("ERROR: No feature cache found.")
        sys.exit(1)

    cache = torch.load(cache_path)
    img_all = cache["img_feats"]
    txt_all = cache["txt_feats"]
    labels_all = cache["labels"]
    train_idx = cache["train_idx"]
    val_idx = cache["val_idx"]
    test_idx = cache["test_idx"]

    train_l = labels_all[train_idx]
    class_counts = torch.bincount(train_l, minlength=NUM_CLASSES).float()

    weights_inv = (1.0 / class_counts)
    weights_inv = weights_inv / weights_inv.sum() * NUM_CLASSES

    weights_sqrt = 1.0 / torch.sqrt(class_counts)
    weights_sqrt = weights_sqrt / weights_sqrt.sum() * NUM_CLASSES

    weights_manual = torch.tensor([4.0, 2.0, 1.0])
    weights_manual = weights_manual / weights_manual.sum() * NUM_CLASSES

    beta = 0.999
    effective = 1.0 - torch.pow(torch.full_like(class_counts, beta), class_counts)
    weights_eff = (1.0 - beta) / torch.clamp(effective, min=1e-8)
    weights_eff = weights_eff / weights_eff.sum() * NUM_CLASSES

    sw_inv = (1.0 / class_counts)[train_l].tolist()
    sampler_inv = WeightedRandomSampler(sw_inv, num_samples=len(train_l), replacement=True)

    sw_sqrt = (1.0 / torch.sqrt(class_counts))[train_l].tolist()
    sampler_sqrt = WeightedRandomSampler(sw_sqrt, num_samples=len(train_l), replacement=True)

    train_ds = FusionDS(img_all[train_idx], txt_all[train_idx], labels_all[train_idx])
    val_ds = FusionDS(img_all[val_idx], txt_all[val_idx], labels_all[val_idx])
    test_ds = FusionDS(img_all[test_idx], txt_all[test_idx], labels_all[test_idx])

    return {
        "class_counts": class_counts,
        "weights": {"inv": weights_inv, "sqrt": weights_sqrt, "manual": weights_manual, "eff": weights_eff},
        "samplers": {"inv": sampler_inv, "sqrt": sampler_sqrt},
        "datasets": {"train": train_ds, "val": val_ds, "test": test_ds},
        "labels_train": train_l,
    }


def make_loaders(data, sampler_key="inv", batch_size=32):
    return (
        DataLoader(data["datasets"]["train"], batch_size=batch_size, sampler=data["samplers"][sampler_key], collate_fn=collate, num_workers=0),
        DataLoader(data["datasets"]["val"], batch_size=64, shuffle=False, collate_fn=collate, num_workers=0),
        DataLoader(data["datasets"]["test"], batch_size=64, shuffle=False, collate_fn=collate, num_workers=0),
    )


# ═══════════════════════════════════════════════════════════════════
# Config batches
# ═══════════════════════════════════════════════════════════════════
def get_batch_configs(batch_num, data):
    cfgs = []
    counts = data["class_counts"].tolist()
    w = data["weights"]

    if batch_num == 0:
        # ── Label smoothing + focal combos ──
        for smoothing in [0.05, 0.1, 0.15, 0.2]:
            for gamma in [1.0, 2.0, 3.0]:
                for greg in [0.5, 1.0]:
                    for lr in [3e-4, 1e-3]:
                        name = f"ls{int(smoothing*100)}_foc{gamma}_g{greg}_lr{lr:.0e}"
                        criterion = LabelSmoothingCE(alpha=w["inv"], smoothing=smoothing)
                        tr, vl, te = make_loaders(data)
                        cfgs.append({"name": name, "model_cls": GatedFusion,
                                     "model_kwargs": {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": FUSION_DROPOUT},
                                     "criterion": criterion, "gate_rw": greg, "lr": lr,
                                     "weight_decay": 1e-4, "patience": 10, "max_epochs": 40,
                                     "scheduler": "warmup_cosine", "feature_noise": 0.0,
                                     "train_loader": tr, "val_loader": vl, "test_loader": te})

    elif batch_num == 1:
        # ── Larger fusion heads ──
        for arch_name, arch_cls, arch_kw in [
            ("wide", GatedFusionWide, {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": 0.3}),
            ("wide_d02", GatedFusionWide, {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": 0.2}),
            ("deep", GatedFusionDeep, {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": 0.3}),
        ]:
            for lr in [1e-4, 3e-4, 1e-3]:
                for greg in [0.5, 1.0]:
                    for loss_name, criterion in [
                        ("wce", WeightedCE(alpha=w["inv"])),
                        ("ls10", LabelSmoothingCE(alpha=w["inv"], smoothing=0.1)),
                    ]:
                        name = f"{arch_name}_{loss_name}_g{greg}_lr{lr:.0e}"
                        tr, vl, te = make_loaders(data)
                        cfgs.append({"name": name, "model_cls": arch_cls,
                                     "model_kwargs": arch_kw,
                                     "criterion": criterion, "gate_rw": greg, "lr": lr,
                                     "weight_decay": 1e-4, "patience": 10, "max_epochs": 40,
                                     "scheduler": "warmup_cosine", "feature_noise": 0.0,
                                     "train_loader": tr, "val_loader": vl, "test_loader": te})

    elif batch_num == 2:
        # ── Class-balanced + manual weights + sqrt ──
        cb_loss = ClassBalancedLoss(counts, beta=0.999, gamma=2.0).to(DEV)
        for loss_name, criterion in [
            ("cb", cb_loss),
            ("wce_manual", WeightedCE(alpha=w["manual"])),
            ("wce_sqrt", WeightedCE(alpha=w["sqrt"])),
            ("wce_eff", WeightedCE(alpha=w["eff"])),
            ("ls_manual", LabelSmoothingCE(alpha=w["manual"], smoothing=0.1)),
            ("ls_sqrt", LabelSmoothingCE(alpha=w["sqrt"], smoothing=0.1)),
        ]:
            for lr in [1e-4, 3e-4, 1e-3]:
                for greg in [0.5, 1.0]:
                    sampler_k = "sqrt" if "sqrt" in loss_name else "inv"
                    name = f"{loss_name}_g{greg}_lr{lr:.0e}"
                    tr, vl, te = make_loaders(data, sampler_key=sampler_k)
                    cfgs.append({"name": name, "model_cls": GatedFusion,
                                 "model_kwargs": {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": FUSION_DROPOUT},
                                 "criterion": criterion, "gate_rw": greg, "lr": lr,
                                 "weight_decay": 1e-4, "patience": 10, "max_epochs": 40,
                                 "scheduler": "warmup_cosine", "feature_noise": 0.0,
                                 "train_loader": tr, "val_loader": vl, "test_loader": te})

    elif batch_num == 3:
        # ── Feature noise + wider head combos ──
        for noise in [0.0, 0.005, 0.01, 0.02]:
            for lr in [3e-4, 1e-3]:
                for greg in [0.5, 1.0]:
                    for wd in [1e-5, 1e-4, 1e-3]:
                        name = f"noise{noise}_g{greg}_lr{lr:.0e}_wd{wd:.0e}"
                        criterion = LabelSmoothingCE(alpha=w["inv"], smoothing=0.1)
                        tr, vl, te = make_loaders(data)
                        cfgs.append({"name": name, "model_cls": GatedFusionWide,
                                     "model_kwargs": {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": 0.3},
                                     "criterion": criterion, "gate_rw": greg, "lr": lr,
                                     "weight_decay": wd, "patience": 10, "max_epochs": 40,
                                     "scheduler": "warmup_cosine", "feature_noise": noise,
                                     "train_loader": tr, "val_loader": vl, "test_loader": te})

    elif batch_num == 4:
        # ── Best of each + ensemble variants ──
        # Repeat v2 best config exactly for reference
        name = "v2_ref_wce_inv_g1.0_lr1e-3"
        tr, vl, te = make_loaders(data)
        cfgs.append({"name": name, "model_cls": GatedFusion,
                     "model_kwargs": {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": FUSION_DROPOUT},
                     "criterion": WeightedCE(alpha=w["inv"]), "gate_rw": 1.0, "lr": 1e-3,
                     "weight_decay": 1e-4, "patience": 10, "max_epochs": 40,
                     "scheduler": "cosine", "feature_noise": 0.0,
                     "train_loader": tr, "val_loader": vl, "test_loader": te})

        # Manual weights, higher gate reg, wide head
        name = "man_wide_g2.0_lr3e-4"
        tr, vl, te = make_loaders(data)
        cfgs.append({"name": name, "model_cls": GatedFusionWide,
                     "model_kwargs": {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": 0.25},
                     "criterion": LabelSmoothingCE(alpha=w["manual"], smoothing=0.1), "gate_rw": 2.0, "lr": 3e-4,
                     "weight_decay": 1e-4, "patience": 15, "max_epochs": 40,
                     "scheduler": "warmup_cosine", "feature_noise": 0.005,
                     "train_loader": tr, "val_loader": vl, "test_loader": te})

        # Eff weights + low gate + low LR
        name = "eff_lowgate_lowlr"
        tr, vl, te = make_loaders(data)
        cfgs.append({"name": name, "model_cls": GatedFusionWide,
                     "model_kwargs": {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": 0.3},
                     "criterion": WeightedCE(alpha=w["eff"]), "gate_rw": 0.5, "lr": 1e-4,
                     "weight_decay": 1e-5, "patience": 15, "max_epochs": 40,
                     "scheduler": "warmup_cosine", "feature_noise": 0.0,
                     "train_loader": tr, "val_loader": vl, "test_loader": te})

        # No gate reg, just WCE, standard architecture
        name = "nogreg_wce_inv_lr1e-3"
        tr, vl, te = make_loaders(data)
        cfgs.append({"name": name, "model_cls": GatedFusion,
                     "model_kwargs": {"img_dim": 1280, "txt_dim": 768, "num_classes": NUM_CLASSES, "dropout": 0.3},
                     "criterion": WeightedCE(alpha=w["inv"]), "gate_rw": 0.0, "lr": 1e-3,
                     "weight_decay": 1e-4, "patience": 10, "max_epochs": 40,
                     "scheduler": "cosine", "feature_noise": 0.0,
                     "train_loader": tr, "val_loader": vl, "test_loader": te})

    return cfgs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    batch_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()
    V3_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  CrisisMind v3 — Batch {batch_num}")
    print("=" * 70)

    data = load_data()
    cfgs = get_batch_configs(batch_num, data)

    print(f"  Configs in this batch: {len(cfgs)}")

    all_summaries = []
    for i, cfg in enumerate(cfgs, 1):
        name = cfg["name"]
        sp = V3_DIR / f"{name}_summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            print(f"\n  [{i}/{len(cfgs)}] {name}: cached (F1={s['f1_macro']:.4f})")
            all_summaries.append(s)
            continue

        print(f"\n  [{i}/{len(cfgs)}] Running: {name}")
        s = train_run(
            name,
            cfg["train_loader"], cfg["val_loader"], cfg["test_loader"],
            cfg["model_cls"], cfg["model_kwargs"],
            cfg["criterion"], cfg["gate_rw"], cfg["lr"],
            cfg["weight_decay"], cfg["patience"], cfg["max_epochs"],
            cfg["scheduler"], cfg["feature_noise"],
        )
        all_summaries.append(s)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  BATCH {batch_num} COMPLETE ({elapsed/60:.1f} min, {len(all_summaries)} runs)")
    print(f"{'='*70}")
    for s in sorted(all_summaries, key=lambda x: -x["f1_macro"]):
        status = "PASS" if (s.get("img_gate", 0) >= 0.10 and s.get("txt_gate", 0) >= 0.10) else "REJECT"
        print(f"  {s['name']:40s}  F1={s['f1_macro']:.4f}  G[I={s['img_gate']:.3f},T={s['txt_gate']:.3f}]  {status}")
    print(f"  v1 baseline: F1=0.5375  |  v2 best: F1=0.5256")


if __name__ == "__main__":
    main()
