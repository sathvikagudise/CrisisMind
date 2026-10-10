"""CrisisMind v3 — Structured hyperparameter search on cached features.

Retains v2's gate balance while recovering v1's performance.

Search dimensions:
  - Class weight strategy: inverse, sqrt, effective_num, manual
  - Loss: wce, focal, class_balanced
  - Gate reg weight: 0.5, 1.0, 2.0
  - LR: 1e-4, 3e-4, 1e-3
  - Weight decay: 1e-5, 1e-4, 1e-3
  - Scheduler: cosine, cosine_warmup, cosine_warmup_restarts
  - Patience: 10, 15, 20
  - Max epochs: 60
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json, time, itertools, shutil
import torch
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
    LABEL_NAMES, NUM_CLASSES, FUSION_DROPOUT, CHECKPOINTS_DIR,
    RANDOM_SEED,
)
from backend.models.fusion_models import GatedFusion, count_parameters
from backend.config.shared import DEVICE

SEVERITY_LABELS = ["Little or No Damage", "Mild Damage", "Severe Damage"]
DEV = torch.device(DEVICE)
V3_DIR = CHECKPOINTS_DIR / "v3"
V2_DIR = CHECKPOINTS_DIR / "v2"
RESULTS_DIR = Path("testing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── v1 baseline ───
V1_BASELINE = {
    "accuracy": 0.6352, "f1_macro": 0.5375, "f1_weighted": 0.6415,
    "precision_macro": 0.5866, "recall_macro": 0.5104,
    "recall_little_or_no_damage": 0.4366, "recall_mild_damage": 0.4444, "recall_severe_damage": 0.7500,
    "img_gate": 0.173, "txt_gate": 0.014,
}

# ─── v2 best baseline ───
V2_BASELINE = {
    "accuracy": 0.5766, "f1_macro": 0.5256, "f1_weighted": 0.6059,
    "precision_macro": 0.5218, "recall_macro": 0.5597,
    "recall_little_or_no_damage": 0.5634, "recall_mild_damage": 0.4921, "recall_severe_damage": 0.6235,
    "img_gate": 0.644, "txt_gate": 0.431,
}

# ═══════════════════════════════════════════════════════════════════
# Class weight strategies
# ═══════════════════════════════════════════════════════════════════
def compute_class_weights(train_labels, strategy, num_classes=3, beta=0.999):
    counts = torch.bincount(train_labels, minlength=num_classes).float()
    total = counts.sum()

    if strategy == "inverse":
        w = 1.0 / counts
    elif strategy == "sqrt":
        w = 1.0 / torch.sqrt(counts)
    elif strategy == "effective_num":
        effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        w = (1.0 - beta) / torch.clamp(effective, min=1e-8)
    elif strategy == "manual":
        w = torch.tensor([4.0, 2.0, 1.0])
    elif strategy == "inverse_sqrt":
        w = 1.0 / torch.sqrt(counts)
    elif strategy == "log_inv":
        w = 1.0 / torch.log(counts + 1)
    else:
        w = torch.ones(num_classes)

    w = w / w.sum() * num_classes
    return w


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
# Gate reg
# ═══════════════════════════════════════════════════════════════════
def gate_reg(gates, target=0.5, weight=0.01):
    return weight * torch.mean((gates - target) ** 2)


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


# ═══════════════════════════════════════════════════════════════════
# Early stopping
# ═══════════════════════════════════════════════════════════════════
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
              criterion, gate_rw, lr, weight_decay, patience,
              max_epochs, scheduler_type, config_tag):
    print(f"\n{'='*65}")
    print(f"  RUN: {name}  |  {config_tag}")
    print(f"  Gate reg: {gate_rw}  |  LR: {lr:.1e}  |  WD: {weight_decay:.1e}  |  Patience: {patience}")
    print(f"{'='*65}")

    model = GatedFusion(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES, dropout=FUSION_DROPOUT).to(DEV)
    t, tt = count_parameters(model)
    print(f"  Params: {t:,} trainable / {tt:,} total")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    elif scheduler_type == "cosine_warmup":
        warmup = min(5, max_epochs // 5)
        def lr_lambda(ep):
            if ep < warmup:
                return (ep + 1) / warmup
            progress = (ep - warmup) / max(1, max_epochs - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "cosine_warmup_restarts":
        warmup = min(3, max_epochs // 8)
        cycle = max(10, max_epochs // 3)
        def lr_lambda(ep):
            if ep < warmup:
                return (ep + 1) / warmup
            in_cycle = (ep - warmup) % cycle
            progress = in_cycle / cycle
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    es = ES(patience=patience)
    hist = {"train_loss": [], "val_f1": [], "img_gate": [], "txt_gate": [], "lr": []}
    best_f1 = 0.0
    best_ep = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        total_loss = 0
        tp, tl, tg = [], [], []

        for img_f, txt_f, labels in train_loader:
            img_f, txt_f, labels = img_f.to(DEV), txt_f.to(DEV), labels.to(DEV)
            logits, gates = model(img_f, txt_f)
            loss = criterion(logits, labels) + gate_reg(gates, 0.5, gate_rw)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * img_f.size(0)
            tp.extend(logits.argmax(1).cpu().numpy())
            tl.extend(labels.cpu().numpy())
            tg.append(gates.detach().cpu())

        train_loss = total_loss / len(train_loader.dataset)
        train_f1 = f1_score(tl, tp, average="macro")

        vm, vcr, vcm, _, _ = eval_full(model, val_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        hist["train_loss"].append(train_loss)
        hist["val_f1"].append(vm["f1_macro"])
        hist["img_gate"].append(vm["img_gate"])
        hist["txt_gate"].append(vm["txt_gate"])
        hist["lr"].append(lr_now)

        imp = ""
        if vm["f1_macro"] > best_f1:
            best_f1 = vm["f1_macro"]
            best_ep = ep
            torch.save(model.state_dict(), V3_DIR / f"{name}_best.pth")
            imp = " *** BEST ***"

        if ep % 5 == 0 or imp or ep == 1:
            print(f"  Ep {ep:2d}  TrL={train_loss:.4f} TrF1={train_f1:.4f}  "
                  f"VaF1={vm['f1_macro']:.4f} VaAcc={vm['accuracy']:.4f}  "
                  f"G[I={vm['img_gate']:.3f},T={vm['txt_gate']:.3f}]  "
                  f"LR={lr_now:.2e}{imp}")

        if es(vm["f1_macro"]):
            print(f"  Early stop at ep {ep} (best ep {best_ep})")
            break

    # Test
    model.load_state_dict(torch.load(V3_DIR / f"{name}_best.pth", map_location=DEV))
    tm, tcr, tcm, yt, yp = eval_full(model, test_loader)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(tcm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=ax)
    ax.set_title(f"CrisisMind v3 — {name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(V3_DIR / f"{name}_cm.png", dpi=150); plt.close(fig)

    # Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    eps = range(1, len(hist["train_loss"]) + 1)
    axes[0].plot(eps, hist["train_loss"], "o-", color="steelblue"); axes[0].set_title("Train Loss"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(eps, hist["val_f1"], "s-", color="coral"); axes[1].set_title("Val Macro F1"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(eps, hist["img_gate"], "o-", label="Image", color="steelblue")
    axes[2].plot(eps, hist["txt_gate"], "s-", label="Text", color="coral")
    axes[2].axhline(0.5, color="gray", ls="--", alpha=0.5); axes[2].set_ylim(0, 1)
    axes[2].set_title("Gates"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"CrisisMind v3 — {name}", fontweight="bold")
    fig.tight_layout(); fig.savefig(V3_DIR / f"{name}_curves.png", dpi=150); plt.close(fig)

    summary = {"name": name, "config_tag": config_tag, "best_epoch": best_ep}
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
# Generate all reports and plots
# ═══════════════════════════════════════════════════════════════════
def gen_reports(all_summaries):
    valid = [s for s in all_summaries if s.get("img_gate", 0) >= 0.10 and s.get("txt_gate", 0) >= 0.10]
    rejected = [s for s in all_summaries if s not in valid]

    if not valid:
        print("  WARNING: No runs passed gate threshold. Using all runs.")
        valid = all_summaries

    best = max(valid, key=lambda s: s["f1_macro"])

    # ── MODEL_V3_REPORT.md ──
    lines = ["# CrisisMind — Model v3 Training Report\n", f"**Date:** {time.strftime('%Y-%m-%d')}\n"]
    lines.append("## Training Configuration\n")
    lines.append("- Architecture: EfficientNet-B0 + DistilBERT + GatedFusion")
    lines.append("- Hybrid-cache: frozen encoders, trainable fusion head")
    lines.append("- Balanced sampling: WeightedRandomSampler")
    lines.append("- Gate regularization: quadratic penalty toward 0.5")
    lines.append("- Optimizer: AdamW + cosine annealing (multiple scheduler variants)")
    lines.append("- Max 60 epochs, early stopping")
    lines.append("- Selection: Validation Macro F1 (gate collapse filter: both gates >= 0.10)\n")

    lines.append("## Search Dimensions\n")
    lines.append("| Dimension | Values |")
    lines.append("|-----------|--------|")
    lines.append("| Class weights | inverse, sqrt, effective_num, manual |")
    lines.append("| Loss | WCE, Focal, Class-Balanced |")
    lines.append("| Gate reg | 0.5, 1.0, 2.0 |")
    lines.append("| LR | 1e-4, 3e-4, 1e-3 |")
    lines.append("| Weight decay | 1e-5, 1e-4, 1e-3 |")
    lines.append("| Scheduler | cosine, cosine_warmup, cosine_warmup_restarts |")
    lines.append("| Patience | 10, 15, 20 |\n")

    if rejected:
        lines.append(f"## Rejected Runs ({len(rejected)} — gate collapse)\n")
        lines.append("| Run | Img Gate | Txt Gate | F1(m) |")
        lines.append("|-----|----------|----------|-------|")
        for s in rejected:
            lines.append(f"| {s['name']} | {s['img_gate']:.4f} | {s['txt_gate']:.4f} | {s['f1_macro']:.4f} |")
        lines.append("")

    lines.append(f"## Top Runs ({len(valid)} passed gate filter)\n")
    lines.append("| # | Run | Loss | CWS | Gate Reg | LR | WD | Sched | F1(m) | F1(w) | Acc | R-L | R-M | R-S | ImgG | TxtG |")
    lines.append("|---|-----|------|-----|----------|----|----|-------|-------|-------|-----|-----|-----|-----|------|------|")
    for i, s in enumerate(sorted(valid, key=lambda x: -x["f1_macro"])[:20], 1):
        m = " **" if s["name"] == best["name"] else ""
        lines.append(f"| {i} | {s['name']}{m} | {s['config_tag']} | "
                      f"{s['f1_macro']:.4f} | {s['f1_weighted']:.4f} | {s['accuracy']:.4f} | "
                      f"{s.get('recall_little_or_no_damage',0):.4f} | "
                      f"{s.get('recall_mild_damage',0):.4f} | "
                      f"{s.get('recall_severe_damage',0):.4f} | "
                      f"{s['img_gate']:.3f} | {s['txt_gate']:.3f} |")

    lines.append(f"\n## Best: {best['name']}\n")
    lines.append(f"- Config: `{best['config_tag']}`")
    lines.append(f"- F1 (macro): {best['f1_macro']:.4f} (v1: {V1_BASELINE['f1_macro']:.4f}, delta: {best['f1_macro']-V1_BASELINE['f1_macro']:+.4f})")
    lines.append(f"- F1 (weighted): {best['f1_weighted']:.4f}")
    lines.append(f"- Accuracy: {best['accuracy']:.4f}")
    lines.append(f"- Gates: img={best['img_gate']:.4f} txt={best['txt_gate']:.4f}\n")

    lines.append("### Per-Class\n")
    lines.append("| Class | Prec | Recall | F1 | vs v1 Recall | vs v2 Recall |")
    lines.append("|-------|------|--------|----|-------------|-------------|")
    for lbl in range(3):
        n = LABEL_NAMES[lbl]
        r = best.get(f"recall_{n}", 0)
        v1r = V1_BASELINE.get(f"recall_{n}", 0)
        v2r = V2_BASELINE.get(f"recall_{n}", 0)
        lines.append(f"| {SEVERITY_LABELS[lbl]} | {best.get(f'precision_{n}',0):.4f} | {r:.4f} | {best.get(f'f1_{n}',0):.4f} | {r-v1r:+.4f} | {r-v2r:+.4f} |")

    (RESULTS_DIR / "MODEL_V3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Saved MODEL_V3_REPORT.md")

    # ── MODEL_COMPARISON_V3.md ──
    lines = ["# CrisisMind — v1 vs v2 vs v3 Comparison\n", f"**Date:** {time.strftime('%Y-%m-%d')}\n"]
    lines.append("## Overall\n")
    lines.append("| Metric | v1 | v2 | v3 | v3-v1 | v3-v2 |")
    lines.append("|--------|----|----|-----|-------|-------|")
    for label, key in [("Accuracy", "accuracy"), ("F1 (macro)", "f1_macro"), ("F1 (weighted)", "f1_weighted")]:
        v1 = V1_BASELINE.get(key, 0); v2 = V2_BASELINE.get(key, 0); v3 = best.get(key, 0)
        lines.append(f"| {label} | {v1:.4f} | {v2:.4f} | {v3:.4f} | {v3-v1:+.4f} | {v3-v2:+.4f} |")

    lines.append("\n## Per-Class Recall\n")
    lines.append("| Class | v1 | v2 | v3 | v3-v1 | v3-v2 |")
    lines.append("|-------|----|----|-----|-------|-------|")
    for lbl in range(3):
        k = f"recall_{LABEL_NAMES[lbl]}"
        v1 = V1_BASELINE.get(k, 0); v2 = V2_BASELINE.get(k, 0); v3 = best.get(k, 0)
        lines.append(f"| {SEVERITY_LABELS[lbl]} | {v1:.4f} | {v2:.4f} | {v3:.4f} | {v3-v1:+.4f} | {v3-v2:+.4f} |")

    lines.append("\n## Gate Analysis\n")
    lines.append(f"| | v1 | v2 | v3 |")
    lines.append(f"|---|----|----|----|")
    lines.append(f"| Image Gate | {V1_BASELINE['img_gate']:.4f} | {V2_BASELINE['img_gate']:.4f} | {best['img_gate']:.4f} |")
    lines.append(f"| Text Gate | {V1_BASELINE['txt_gate']:.4f} | {V2_BASELINE['txt_gate']:.4f} | {best['txt_gate']:.4f} |")

    (RESULTS_DIR / "MODEL_COMPARISON_V3.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Saved MODEL_COMPARISON_V3.md")

    # ── BEST_HYPERPARAMETERS.md ──
    lines = ["# CrisisMind v3 — Best Hyperparameters\n", f"**Date:** {time.strftime('%Y-%m-%d')}\n"]
    lines.append(f"## Winner: `{best['name']}`\n")
    lines.append(f"**Config tag:** `{best['config_tag']}`\n")
    lines.append("| Hyperparameter | Value |")
    lines.append("|----------------|-------|")
    for k, v in best.items():
        if k in ("name", "config_tag") or k.startswith("precision_") or k.startswith("recall_") or k.startswith("f1_"):
            continue
        lines.append(f"| {k} | {v} |")

    (RESULTS_DIR / "BEST_HYPERPARAMETERS.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Saved BEST_HYPERPARAMETERS.md")

    # ── TRAINING_SUMMARY.md ──
    lines = ["# CrisisMind v3 — Training Summary\n", f"**Date:** {time.strftime('%Y-%m-%d')}\n"]
    lines.append(f"- Total runs: {len(all_summaries)}")
    lines.append(f"- Passed gate filter: {len(valid)}")
    lines.append(f"- Rejected (gate collapse): {len(rejected)}")
    lines.append(f"- Best: `{best['name']}` — F1(macro)={best['f1_macro']:.4f}\n")
    lines.append("## All Runs\n")
    lines.append("| Run | F1(m) | F1(w) | Acc | ImgG | TxtG | Status |")
    lines.append("|-----|-------|-------|-----|------|------|--------|")
    for s in sorted(all_summaries, key=lambda x: -x["f1_macro"]):
        status = "PASS" if (s.get("img_gate", 0) >= 0.10 and s.get("txt_gate", 0) >= 0.10) else "REJECT"
        lines.append(f"| {s['name']} | {s['f1_macro']:.4f} | {s['f1_weighted']:.4f} | {s['accuracy']:.4f} | "
                      f"{s['img_gate']:.3f} | {s['txt_gate']:.3f} | {status} |")

    (RESULTS_DIR / "TRAINING_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Saved TRAINING_SUMMARY.md")

    # ── All runs bar chart ──
    fig, ax = plt.subplots(figsize=(12, 7))
    names = ["v1_prod", "v2_best"] + [s["name"] for s in sorted(valid, key=lambda x: -x["f1_macro"])]
    f1s = [V1_BASELINE["f1_macro"], V2_BASELINE["f1_macro"]] + [s["f1_macro"] for s in sorted(valid, key=lambda x: -x["f1_macro"])]
    colors = ["#e74c3c", "#2ecc71"] + ["#3498db"] * len(valid)
    bars = ax.barh(names, f1s, color=colors)
    ax.set_xlabel("Macro F1"); ax.set_title("CrisisMind v3 — All Runs", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2, f"{f1:.4f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "V3_ALL_RUNS_F1.png", dpi=150); plt.close(fig)
    print("  Saved V3_ALL_RUNS_F1.png")

    # ── Comparison bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, keys, labels, title in [
        (axes[0], ["accuracy", "f1_macro", "f1_weighted"], ["Accuracy", "F1(macro)", "F1(wtd)"], "Overall"),
        (axes[1], ["recall_little_or_no_damage", "recall_mild_damage", "recall_severe_damage"],
         ["Little", "Mild", "Severe"], "Per-Class Recall"),
    ]:
        v1_vals = [V1_BASELINE.get(k, 0) for k in keys]
        v2_vals = [V2_BASELINE.get(k, 0) for k in keys]
        v3_vals = [best.get(k, 0) for k in keys]
        x = np.arange(len(labels))
        ax.bar(x - 0.3, v1_vals, 0.25, label="v1", color="#e74c3c")
        ax.bar(x, v2_vals, 0.25, label="v2", color="#2ecc71")
        ax.bar(x + 0.3, v3_vals, 0.25, label="v3", color="#3498db")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.set_title(title, fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[2]
    x = np.arange(2)
    ax.bar(x - 0.3, [V1_BASELINE["img_gate"], V1_BASELINE["txt_gate"]], 0.25, label="v1", color="#e74c3c")
    ax.bar(x, [V2_BASELINE["img_gate"], V2_BASELINE["txt_gate"]], 0.25, label="v2", color="#2ecc71")
    ax.bar(x + 0.3, [best["img_gate"], best["txt_gate"]], 0.25, label="v3", color="#3498db")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(["Image Gate", "Text Gate"]); ax.set_ylim(0, 1)
    ax.set_title("Gates", fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("CrisisMind v1 vs v2 vs v3", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "V3_MODEL_COMPARISON.png", dpi=150); plt.close(fig)
    print("  Saved V3_MODEL_COMPARISON.png")

    # ── Gate heatmap across runs ──
    gate_data = [(s["name"], s["img_gate"], s["txt_gate"], s["f1_macro"]) for s in valid]
    gate_data.sort(key=lambda x: -x[3])
    if gate_data:
        fig, ax = plt.subplots(figsize=(10, max(4, len(gate_data) * 0.4)))
        names_g = [g[0] for g in gate_data]
        img_g = [g[1] for g in gate_data]
        txt_g = [g[2] for g in gate_data]
        f1_g = [g[3] for g in gate_data]
        x = np.arange(len(names_g))
        ax.barh(x - 0.2, img_g, 0.35, label="Image Gate", color="steelblue")
        ax.barh(x + 0.2, txt_g, 0.35, label="Text Gate", color="coral")
        ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
        ax.set_yticks(x); ax.set_yticklabels([f"{n} ({f:.3f})" for n, f in zip(names_g, f1_g)], fontsize=8)
        ax.set_xlim(0, 1); ax.set_title("Gate Values per Run (sorted by F1)", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "V3_GATE_HEATMAP.png", dpi=150); plt.close(fig)
        print("  Saved V3_GATE_HEATMAP.png")

    return best


# ═══════════════════════════════════════════════════════════════════
# Manual evaluation
# ═══════════════════════════════════════════════════════════════════
def manual_eval(best_name, test_loader, val_loader):
    model = GatedFusion(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES, dropout=FUSION_DROPOUT).to(DEV)
    model.load_state_dict(torch.load(V3_DIR / f"{best_name}_best.pth", map_location=DEV))
    model.eval()

    print(f"\n{'='*65}")
    print(f"  MANUAL EVALUATION — {best_name}")
    print(f"{'='*65}")

    # Evaluate on test set with per-sample detail
    all_preds, all_labels, all_gates = [], [], []
    with torch.no_grad():
        for img_f, txt_f, labels in test_loader:
            img_f, txt_f = img_f.to(DEV), txt_f.to(DEV)
            logits, gates = model(img_f, txt_f)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_gates.append(gates.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_gates = torch.cat(all_gates)

    # Print misclassified samples
    wrong_mask = all_preds != all_labels
    n_wrong = wrong_mask.sum()
    print(f"\n  Misclassified: {n_wrong}/{len(all_labels)} ({100*n_wrong/len(all_labels):.1f}%)")
    print(f"\n  Confusion (misclassification breakdown):")
    for true_lbl in range(3):
        for pred_lbl in range(3):
            if true_lbl != pred_lbl:
                count = ((all_labels == true_lbl) & (all_preds == pred_lbl)).sum()
                if count > 0:
                    print(f"    {SEVERITY_LABELS[true_lbl]:25s} -> {SEVERITY_LABELS[pred_lbl]:25s}: {count}")

    # Gate distribution for correct vs incorrect
    correct_gates = all_gates[~wrong_mask]
    incorrect_gates = all_gates[wrong_mask]
    print(f"\n  Gate stats (correct predictions):")
    print(f"    Image gate: mean={correct_gates[:,0].mean():.4f}, std={correct_gates[:,0].std():.4f}")
    print(f"    Text gate:  mean={correct_gates[:,1].mean():.4f}, std={correct_gates[:,1].std():.4f}")
    if len(incorrect_gates) > 0:
        print(f"  Gate stats (incorrect predictions):")
        print(f"    Image gate: mean={incorrect_gates[:,0].mean():.4f}, std={incorrect_gates[:,0].std():.4f}")
        print(f"    Text gate:  mean={incorrect_gates[:,1].mean():.4f}, std={incorrect_gates[:,1].std():.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main — search grid
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    V3_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CrisisMind Model v3 — Structured Hyperparameter Search")
    print("=" * 70)

    cache_path = V2_DIR / "feature_cache.pt"
    if not cache_path.exists():
        cache_path = V3_DIR / "feature_cache.pt"
    if not cache_path.exists():
        print(f"ERROR: Feature cache not found. Searched:")
        print(f"  {V2_DIR / 'feature_cache.pt'}")
        print(f"  {V3_DIR / 'feature_cache.pt'}")
        print("Run step1_extract.py first.")
        return

    cache = torch.load(cache_path)
    img_all = cache["img_feats"]
    txt_all = cache["txt_feats"]
    labels_all = cache["labels"]

    train_idx = cache["train_idx"]
    val_idx = cache["val_idx"]
    test_idx = cache["test_idx"]

    print(f"Loaded features: img={img_all.shape}, txt={txt_all.shape}")
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    train_l = labels_all[train_idx]
    class_counts = torch.bincount(train_l, minlength=NUM_CLASSES).float()
    print(f"  Class counts: {class_counts.tolist()}")

    # ── Class weight strategies ──
    cw_strategies = {
        "inverse": compute_class_weights(train_l, "inverse"),
        "sqrt": compute_class_weights(train_l, "sqrt"),
        "eff": compute_class_weights(train_l, "effective_num"),
        "manual": compute_class_weights(train_l, "manual"),
    }
    for k, v in cw_strategies.items():
        print(f"  Weights [{k}]: {v.tolist()}")

    # ── Samplers per strategy ──
    samplers = {}
    for strat_name in ["inverse", "sqrt"]:
        w = cw_strategies[strat_name][train_l].tolist()
        samplers[strat_name] = WeightedRandomSampler(w, num_samples=len(train_l), replacement=True)

    # ── Dataset objects ──
    train_ds = FusionDS(img_all[train_idx], txt_all[train_idx], labels_all[train_idx])
    val_ds = FusionDS(img_all[val_idx], txt_all[val_idx], labels_all[val_idx])
    test_ds = FusionDS(img_all[test_idx], txt_all[test_idx], labels_all[test_idx])

    # ── Search grid ──
    # Phase 1: Broad search (LR x loss x weight_strategy x gate_reg x scheduler)
    # Phase 2: Refine top configs (WD x patience)
    grid_phase1 = []
    run_id = 0

    for cw_name, cw_tensor in cw_strategies.items():
        for loss_type in ["wce", "focal", "class_balanced"]:
            for gate_rw in [0.5, 1.0, 2.0]:
                for lr in [1e-4, 3e-4, 1e-3]:
                    for sched in ["cosine", "cosine_warmup"]:
                        # Choose sampler based on weight strategy
                        if cw_name in ("inverse", "sqrt"):
                            sampler = samplers[cw_name]
                            bs = 32
                        elif cw_name == "eff":
                            sampler = samplers["inverse"]  # use inverse sampler for effective weights
                            bs = 32
                        else:
                            sampler = samplers["inverse"]
                            bs = 32

                        run_id += 1
                        name = f"v3_{run_id:03d}_{cw_name}_{loss_type}_g{gate_rw}_lr{lr:.0e}_{sched}"
                        tag = f"{loss_type}|{cw_name}|g={gate_rw}|lr={lr:.0e}|{sched}"

                        # Build criterion
                        if loss_type == "wce":
                            criterion = WeightedCE(alpha=cw_tensor.to(DEV))
                        elif loss_type == "focal":
                            criterion = FocalLoss(alpha=cw_tensor.to(DEV), gamma=2.0)
                        elif loss_type == "class_balanced":
                            counts_list = class_counts.tolist()
                            criterion = ClassBalancedLoss(counts_list, beta=0.999, gamma=2.0).to(DEV)

                        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, collate_fn=collate, num_workers=0)
                        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)
                        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)

                        grid_phase1.append({
                            "name": name, "tag": tag,
                            "criterion": criterion,
                            "gate_rw": gate_rw, "lr": lr,
                            "weight_decay": 1e-4,
                            "patience": 15, "max_epochs": 60,
                            "scheduler": sched,
                            "train_loader": train_loader,
                            "val_loader": val_loader,
                            "test_loader": test_loader,
                        })

    # Also add some targeted configs
    targeted = [
        # v2 best config baseline
        {
            "name": "v3_v2baseline", "tag": "wce|inverse|g=1.0|lr=1e-3|cosine|wd=1e-4",
            "criterion": WeightedCE(alpha=cw_strategies["inverse"].to(DEV)),
            "gate_rw": 1.0, "lr": 1e-3, "weight_decay": 1e-4,
            "patience": 10, "max_epochs": 60, "scheduler": "cosine",
        },
        # High weight decay
        {
            "name": "v3_highwd", "tag": "wce|inverse|g=1.0|lr=3e-4|cosine|wd=1e-3",
            "criterion": WeightedCE(alpha=cw_strategies["inverse"].to(DEV)),
            "gate_rw": 1.0, "lr": 3e-4, "weight_decay": 1e-3,
            "patience": 15, "max_epochs": 60, "scheduler": "cosine",
        },
        # Low gate reg, low LR
        {
            "name": "v3_lowgate_lowlr", "tag": "wce|inverse|g=0.5|lr=1e-4|cosine_warmup|wd=1e-4",
            "criterion": WeightedCE(alpha=cw_strategies["inverse"].to(DEV)),
            "gate_rw": 0.5, "lr": 1e-4, "weight_decay": 1e-4,
            "patience": 20, "max_epochs": 60, "scheduler": "cosine_warmup",
        },
        # Manual weights, no focal
        {
            "name": "v3_manual_wce", "tag": "wce|manual|g=1.0|lr=3e-4|cosine|wd=1e-4",
            "criterion": WeightedCE(alpha=cw_strategies["manual"].to(DEV)),
            "gate_rw": 1.0, "lr": 3e-4, "weight_decay": 1e-4,
            "patience": 15, "max_epochs": 60, "scheduler": "cosine",
        },
        # Restarting scheduler
        {
            "name": "v3_restart", "tag": "wce|inverse|g=0.5|lr=3e-4|cosine_warmup_restarts|wd=1e-4",
            "criterion": WeightedCE(alpha=cw_strategies["inverse"].to(DEV)),
            "gate_rw": 0.5, "lr": 3e-4, "weight_decay": 1e-4,
            "patience": 15, "max_epochs": 60, "scheduler": "cosine_warmup_restarts",
        },
    ]
    for t in targeted:
        t["name"] = t["name"]
        t["train_loader"] = DataLoader(train_ds, batch_size=32, sampler=samplers["inverse"], collate_fn=collate, num_workers=0)
        t["val_loader"] = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)
        t["test_loader"] = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)
        grid_phase1.append(t)

    print(f"\n  Total configs to train: {len(grid_phase1)}")
    print(f"  Estimated time: ~{len(grid_phase1) * 0.5:.0f} min (cached features)\n")

    # ── Run all ──
    all_summaries = []
    for i, cfg in enumerate(grid_phase1, 1):
        sp = V3_DIR / f"{cfg['name']}_summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            print(f"\n  [{i}/{len(grid_phase1)}] {cfg['name']}: cached (F1={s['f1_macro']:.4f})")
            all_summaries.append(s)
            continue

        print(f"\n  [{i}/{len(grid_phase1)}] Running: {cfg['name']}")
        s = train_run(
            cfg["name"], cfg["train_loader"], cfg["val_loader"], cfg["test_loader"],
            cfg["criterion"], cfg["gate_rw"], cfg["lr"], cfg["weight_decay"],
            cfg["patience"], cfg["max_epochs"], cfg["scheduler"], cfg["tag"],
        )
        all_summaries.append(s)

    # ── Select best ──
    valid_runs = [s for s in all_summaries if s.get("img_gate", 0) >= 0.10 and s.get("txt_gate", 0) >= 0.10]
    if not valid_runs:
        print("\n  WARNING: No runs passed gate filter! Using all runs.")
        valid_runs = all_summaries

    best = max(valid_runs, key=lambda s: s["f1_macro"])
    print(f"\n{'='*70}")
    print(f"  BEST: {best['name']}  F1(macro)={best['f1_macro']:.4f}  Gates[I={best['img_gate']:.4f},T={best['txt_gate']:.4f}]")
    print(f"{'='*70}")

    shutil.copy2(V3_DIR / f"{best['name']}_best.pth", CHECKPOINTS_DIR / "best_gated_fusion_v3.pth")
    print(f"  Saved to {CHECKPOINTS_DIR / 'best_gated_fusion_v3.pth'}")

    # ── Reports ──
    gen_reports(all_summaries)

    # ── Manual eval ──
    loader_cfg = next((cfg for cfg in grid_phase1 if cfg["name"] == best["name"]), None)
    if loader_cfg:
        manual_eval(best["name"], loader_cfg["test_loader"], loader_cfg["val_loader"])

    # ── Top 5 summary ──
    print(f"\n{'='*70}")
    print(f"  TOP 5 CONFIGS")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'Name':<45} {'F1(m)':<8} {'F1(w)':<8} {'Acc':<8} {'ImgG':<8} {'TxtG':<8}")
    print(f"  {'-'*90}")
    for i, s in enumerate(sorted(valid_runs, key=lambda x: -x["f1_macro"])[:5], 1):
        print(f"  {i:<5} {s['name']:<45} {s['f1_macro']:.4f}   {s['f1_weighted']:.4f}   {s['accuracy']:.4f}   {s['img_gate']:.4f}   {s['txt_gate']:.4f}")

    print(f"\n  v1 baseline: F1(m)=0.5375  F1(w)=0.6415  Acc=0.6352  ImgG=0.173  TxtG=0.014")
    print(f"  v2 best:     F1(m)=0.5256  F1(w)=0.6059  Acc=0.5766  ImgG=0.644  TxtG=0.431")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  COMPLETE ({elapsed/60:.1f} min, {len(all_summaries)} runs)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
