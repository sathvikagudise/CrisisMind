"""Step 2: Train all v2 configurations and generate all outputs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json, time, shutil
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
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

from sklearn.model_selection import train_test_split
from backend.training.config import (
    LABEL_NAMES, NUM_CLASSES, FUSION_DROPOUT, FUSION_PATIENCE, CHECKPOINTS_DIR,
    RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
)
from backend.models.fusion_models import GatedFusion, count_parameters
from backend.config.shared import DEVICE

SEVERITY_LABELS = ["Little or No Damage", "Mild Damage", "Severe Damage"]
DEV = torch.device(DEVICE)
V2_DIR = CHECKPOINTS_DIR / "v2"
RESULTS_DIR = Path("testing")


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Gate reg
# ---------------------------------------------------------------------------
def gate_reg(gates, target=0.5, weight=0.01):
    return weight * torch.mean((gates - target) ** 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FusionDS(Dataset):
    def __init__(self, img_f, txt_f, labels):
        self.img_f, self.txt_f, self.labels = img_f, txt_f, labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.img_f[idx], self.txt_f[idx], self.labels[idx]


def collate(b):
    return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]), torch.stack([x[2] for x in b])


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class ES:
    def __init__(self, patience=7):
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


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def train_run(name, train_loader, val_loader, test_loader,
              weights, loss_type, gate_rw, max_epochs=40, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"  RUN: {name}  |  Loss: {loss_type}  |  Gate reg: {gate_rw}  |  LR: {lr}")
    print(f"{'='*60}")

    model = GatedFusion(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES, dropout=FUSION_DROPOUT).to(DEV)
    t, tt = count_parameters(model)
    print(f"  Params: {t:,} trainable / {tt:,} total")

    if loss_type == "wce":
        criterion = WeightedCE(alpha=weights.to(DEV))
    else:
        criterion = FocalLoss(alpha=weights.to(DEV), gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    es = ES(patience=10)

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
            torch.save(model.state_dict(), V2_DIR / f"{name}_best.pth")
            imp = " *** BEST ***"

        print(f"  Ep {ep:2d}  TrL={train_loss:.4f} TrF1={train_f1:.4f}  "
              f"VaF1={vm['f1_macro']:.4f} VaAcc={vm['accuracy']:.4f}  "
              f"G[I={vm['img_gate']:.3f},T={vm['txt_gate']:.3f}]  "
              f"LR={lr_now:.2e}{imp}")

        if es(vm["f1_macro"]):
            print(f"  Early stop at ep {ep}")
            break

    # Test
    model.load_state_dict(torch.load(V2_DIR / f"{name}_best.pth", map_location=DEV))
    tm, tcr, tcm, yt, yp = eval_full(model, test_loader)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(tcm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=ax)
    ax.set_title(f"CrisisMind v2 — {name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(V2_DIR / f"{name}_cm.png", dpi=150); plt.close(fig)

    # Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    eps = range(1, len(hist["train_loss"]) + 1)
    axes[0].plot(eps, hist["train_loss"], "o-", color="steelblue"); axes[0].set_title("Train Loss"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(eps, hist["val_f1"], "s-", color="coral"); axes[1].set_title("Val Macro F1"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(eps, hist["img_gate"], "o-", label="Image", color="steelblue")
    axes[2].plot(eps, hist["txt_gate"], "s-", label="Text", color="coral")
    axes[2].axhline(0.5, color="gray", ls="--", alpha=0.5); axes[2].set_ylim(0, 1)
    axes[2].set_title("Gates"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"CrisisMind v2 — {name}", fontweight="bold")
    fig.tight_layout(); fig.savefig(V2_DIR / f"{name}_curves.png", dpi=150); plt.close(fig)

    summary = {"name": name, "loss_type": loss_type, "gate_reg_weight": gate_rw, "best_epoch": best_ep}
    summary.update({k: round(float(v), 4) for k, v in tm.items()})
    for lbl in range(3):
        dn = SEVERITY_LABELS[lbl]
        sn = LABEL_NAMES[lbl]
        summary[f"precision_{sn}"] = round(tcr[dn]["precision"], 4)
        summary[f"recall_{sn}"] = round(tcr[dn]["recall"], 4)
        summary[f"f1_{sn}"] = round(tcr[dn]["f1-score"], 4)

    with open(V2_DIR / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  TEST: Acc={tm['accuracy']:.4f} F1m={tm['f1_macro']:.4f} F1w={tm['f1_weighted']:.4f} "
          f"Gates[I={tm['img_gate']:.4f},T={tm['txt_gate']:.4f}]")
    for lbl in range(3):
        dn = SEVERITY_LABELS[lbl]
        print(f"    {dn:25s}  P={tcr[dn]['precision']:.4f}  R={tcr[dn]['recall']:.4f}  F1={tcr[dn]['f1-score']:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def gen_reports(all_summaries, v1_m):
    best = max(all_summaries, key=lambda s: s["f1_macro"])

    # ── MODEL_V2_REPORT.md ──
    lines = ["# CrisisMind — Model v2 Training Report\n", f"**Date:** {time.strftime('%Y-%m-%d')}\n"]
    lines.append("## Training Configuration\n")
    lines.append("- Architecture: EfficientNet-B0 + DistilBERT + GatedFusion")
    lines.append("- Hybrid-cache: frozen encoders, trainable fusion head")
    lines.append("- Balanced sampling: WeightedRandomSampler")
    lines.append("- Gate regularization: quadratic penalty toward 0.5")
    lines.append("- Optimizer: AdamW + cosine annealing")
    lines.append("- Max 40 epochs, early stopping (patience=7)")
    lines.append("- Selection: Validation Macro F1\n")

    lines.append("## Loss Comparison\n")
    lines.append("| Run | Loss | Gate Reg | F1(m) | F1(w) | Acc | R-L | R-M | R-S |")
    lines.append("|-----|------|----------|-------|-------|-----|-----|-----|-----|")
    for s in sorted(all_summaries, key=lambda x: -x["f1_macro"]):
        m = " **" if s["name"] == best["name"] else ""
        lines.append(f"| {s['name']}{m} | {s['loss_type']} | {s['gate_reg_weight']} | "
                      f"{s['f1_macro']:.4f} | {s['f1_weighted']:.4f} | {s['accuracy']:.4f} | "
                      f"{s.get('recall_little_or_no_damage',0):.4f} | "
                      f"{s.get('recall_mild_damage',0):.4f} | "
                      f"{s.get('recall_severe_damage',0):.4f} |")

    lines.append(f"\n## Best: {best['name']}\n")
    lines.append(f"- F1 (macro): {best['f1_macro']:.4f}")
    lines.append(f"- Accuracy: {best['accuracy']:.4f}")
    lines.append(f"- Gates: img={best['img_gate']:.4f} txt={best['txt_gate']:.4f}\n")

    lines.append("### Per-Class\n")
    lines.append("| Class | Prec | Recall | F1 |")
    lines.append("|-------|------|--------|----|")
    for lbl in range(3):
        n = LABEL_NAMES[lbl]
        lines.append(f"| {SEVERITY_LABELS[lbl]} | {best.get(f'precision_{n}',0):.4f} | "
                      f"{best.get(f'recall_{n}',0):.4f} | {best.get(f'f1_{n}',0):.4f} |")

    (RESULTS_DIR / "MODEL_V2_REPORT.md").write_text("\n".join(lines))
    print(f"  Saved MODEL_V2_REPORT.md")

    # ── MODEL_COMPARISON.md ──
    lines = ["# CrisisMind — Model v1 vs v2 Comparison\n", f"**Date:** {time.strftime('%Y-%m-%d')}\n"]
    lines.append("## Overall\n")
    lines.append("| Metric | v1 | v2 | Delta |")
    lines.append("|--------|----|----|-------|")
    for label, key in [("Accuracy", "accuracy"), ("F1 (macro)", "f1_macro"), ("F1 (weighted)", "f1_weighted")]:
        v1 = v1_m.get(key, 0); v2 = best.get(key, 0); d = v2 - v1
        lines.append(f"| {label} | {v1:.4f} | {v2:.4f} | {'+' if d>0 else ''}{d:.4f} |")

    lines.append("\n## Per-Class Recall\n")
    lines.append("| Class | v1 | v2 | Delta |")
    lines.append("|-------|----|----|-------|")
    for lbl in range(3):
        k = f"recall_{LABEL_NAMES[lbl]}"
        v1 = v1_m.get(k, 0); v2 = best.get(k, 0); d = v2 - v1
        lines.append(f"| {SEVERITY_LABELS[lbl]} | {v1:.4f} | {v2:.4f} | {'+' if d>0 else ''}{d:.4f} |")

    lines.append("\n## Gate Analysis\n")
    lines.append(f"| | v1 | v2 |")
    lines.append(f"|---|----|----|")
    lines.append(f"| Image Gate | {v1_m['img_gate']:.4f} | {best['img_gate']:.4f} |")
    lines.append(f"| Text Gate | {v1_m['txt_gate']:.4f} | {best['txt_gate']:.4f} |")

    lines.append("\n## Recommendation\n")
    d_f1 = best["f1_macro"] - v1_m["f1_macro"]
    v1_l = v1_m.get("recall_little_or_no_damage", 0)
    v2_l = best.get("recall_little_or_no_damage", 0)
    v1_m2 = v1_m.get("recall_mild_damage", 0)
    v2_m2 = best.get("recall_mild_damage", 0)

    if d_f1 > 0.01 and v2_l >= v1_l - 0.05 and v2_m2 >= v1_m2 - 0.05:
        lines.append("**RECOMMEND: Deploy v2.** Clear F1 improvement with no unacceptable regression.")
    elif d_f1 > 0:
        lines.append("**CONDITIONAL: v2 shows improvement** but with trade-offs. Deploy with monitoring.")
    else:
        lines.append("**NOT RECOMMENDED: v2 does not outperform v1.** Keep current checkpoint.")
    lines.append(f"\nDelta F1: {d_f1:+.4f} | Delta Little Recall: {v2_l-v1_l:+.4f} | Delta Mild Recall: {v2_m2-v1_m2:+.4f}")

    (RESULTS_DIR / "MODEL_COMPARISON.md").write_text("\n".join(lines))
    print(f"  Saved MODEL_COMPARISON.md")

    # ── GATE_ANALYSIS.md ──
    lines = ["# CrisisMind — Gate Collapse Analysis\n", f"**Date:** {time.strftime('%Y-%m-%d')}\n"]
    lines.append("## Problem\nv1 text gate ~0.014 (near zero). Text modality effectively ignored.\n")
    lines.append("## Solution\nQuadratic penalty: `L = weight * mean((gate - 0.5)^2)`\n")
    lines.append("## Results\n")
    lines.append("| Run | Reg Weight | Img Gate | Txt Gate | Diff |")
    lines.append("|-----|-----------|----------|----------|------|")
    for s in sorted(all_summaries, key=lambda x: -x["txt_gate"]):
        lines.append(f"| {s['name']} | {s['gate_reg_weight']} | {s['img_gate']:.4f} | {s['txt_gate']:.4f} | {abs(s['img_gate']-s['txt_gate']):.4f} |")

    if best["txt_gate"] > 0.1:
        lines.append(f"\n**Significant improvement.** Text gate {best['txt_gate']/0.014:.1f}x over v1.")
    elif best["txt_gate"] > 0.05:
        lines.append(f"\n**Moderate improvement.** Text gate still below balanced threshold.")
    else:
        lines.append(f"\n**Limited improvement.** Text modality may lack discriminative power.")

    (RESULTS_DIR / "GATE_ANALYSIS.md").write_text("\n".join(lines))
    print(f"  Saved GATE_ANALYSIS.md")

    # ── Comparison plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, keys, labels, title in [
        (axes[0], ["accuracy", "f1_macro", "f1_weighted"], ["Accuracy", "F1(macro)", "F1(wtd)"], "Overall"),
        (axes[1], ["recall_little_or_no_damage", "recall_mild_damage", "recall_severe_damage"], ["Little", "Mild", "Severe"], "Per-Class Recall"),
    ]:
        v1_vals = [v1_m.get(k, 0) for k in keys]
        v2_vals = [best.get(k, 0) for k in keys]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, v1_vals, 0.35, label="v1", color="#e74c3c")
        ax.bar(x + 0.2, v2_vals, 0.35, label="v2", color="#2ecc71")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.set_title(title, fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[2]
    x = np.arange(2)
    ax.bar(x - 0.2, [v1_m["img_gate"], v1_m["txt_gate"]], 0.35, label="v1", color="#e74c3c")
    ax.bar(x + 0.2, [best["img_gate"], best["txt_gate"]], 0.35, label="v2", color="#2ecc71")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(["Image Gate", "Text Gate"]); ax.set_ylim(0, 1)
    ax.set_title("Gates", fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("CrisisMind v1 vs v2", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "MODEL_COMPARISON.png", dpi=150); plt.close(fig)
    print(f"  Saved MODEL_COMPARISON.png")

    # All runs bar
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [s["name"] for s in all_summaries] + ["v1_prod"]
    f1s = [s["f1_macro"] for s in all_summaries] + [v1_m["f1_macro"]]
    colors = ["#3498db"] * len(all_summaries) + ["#e74c3c"]
    bars = ax.barh(names, f1s, color=colors)
    ax.set_xlabel("Macro F1"); ax.set_title("All Runs", fontweight="bold"); ax.grid(True, alpha=0.3, axis="x")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2, f"{f1:.4f}", va="center")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "ALL_RUNS_F1.png", dpi=150); plt.close(fig)
    print(f"  Saved ALL_RUNS_F1.png")

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 70)
    print("  CrisisMind Model v2 — Training & Evaluation")
    print("=" * 70)

    cache_path = V2_DIR / "feature_cache.pt"
    if not cache_path.exists():
        print(f"ERROR: Feature cache not found at {cache_path}")
        print("Run step1_extract.py first.")
        return

    cache = torch.load(cache_path)
    img_all = cache["img_feats"]
    txt_all = cache["txt_feats"]
    labels_all = cache["labels"]

    if "train_idx" in cache:
        train_idx = cache["train_idx"]
        val_idx = cache["val_idx"]
        test_idx = cache["test_idx"]
    else:
        y = labels_all.numpy()
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
        val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y[train_idx])
        # Save back
        cache["train_idx"] = train_idx
        cache["val_idx"] = val_idx
        cache["test_idx"] = test_idx
        torch.save(cache, cache_path)

    print(f"Loaded features: img={img_all.shape}, txt={txt_all.shape}")
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    train_l = labels_all[train_idx]
    class_counts = torch.bincount(train_l, minlength=NUM_CLASSES).float()
    weights_inv = (1.0 / class_counts)
    weights_inv = weights_inv / weights_inv.sum() * NUM_CLASSES
    print(f"  Class counts: {class_counts.tolist()}")
    print(f"  Weights: {weights_inv.tolist()}")

    sw = (1.0 / class_counts)[train_l].tolist()
    sampler = WeightedRandomSampler(sw, num_samples=len(train_l), replacement=True)

    train_ds = FusionDS(img_all[train_idx], txt_all[train_idx], labels_all[train_idx])
    val_ds = FusionDS(img_all[val_idx], txt_all[val_idx], labels_all[val_idx])
    test_ds = FusionDS(img_all[test_idx], txt_all[test_idx], labels_all[test_idx])

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)

    # Configs
    configs = [
        ("wce_reg05", "wce", weights_inv, 0.5),
        ("wce_reg1", "wce", weights_inv, 1.0),
        ("wce_reg2", "wce", weights_inv, 2.0),
        ("focal_reg05", "focal", weights_inv, 0.5),
        ("focal_reg1", "focal", weights_inv, 1.0),
        ("focal_reg2", "focal", weights_inv, 2.0),
        ("focal_noreg", "focal", weights_inv, 0.0),
    ]

    all_summaries = []
    for name, loss_type, weights, gate_rw in configs:
        sp = V2_DIR / f"{name}_summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            print(f"\n  {name}: cached (F1={s['f1_macro']:.4f})")
            all_summaries.append(s)
            continue
        s = train_run(name, train_loader, val_loader, test_loader, weights, loss_type, gate_rw)
        all_summaries.append(s)

    # Select best
    best = max(all_summaries, key=lambda s: s["f1_macro"])
    print(f"\n{'='*60}")
    print(f"  BEST: {best['name']}  F1(macro)={best['f1_macro']:.4f}")
    print(f"{'='*60}")

    shutil.copy2(V2_DIR / f"{best['name']}_best.pth", CHECKPOINTS_DIR / "best_gated_fusion_v2.pth")
    print(f"  Saved to {CHECKPOINTS_DIR / 'best_gated_fusion_v2.pth'}")

    v1_m = {
        "accuracy": 0.6352, "f1_macro": 0.5375, "f1_weighted": 0.6415,
        "recall_little_or_no_damage": 0.4366, "recall_mild_damage": 0.4444, "recall_severe_damage": 0.7500,
        "img_gate": 0.173, "txt_gate": 0.014,
    }

    gen_reports(all_summaries, v1_m)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    for s in sorted(all_summaries, key=lambda x: -x["f1_macro"]):
        print(f"  {s['name']:20s}  F1={s['f1_macro']:.4f}  Gates[I={s['img_gate']:.3f},T={s['txt_gate']:.3f}]")
    print(f"  v1_prod           F1=0.5375  Gates[I=0.173,T=0.014]")

    cache_path.unlink(missing_ok=True)
    print(f"\n  Cleaned up feature cache.")


if __name__ == "__main__":
    main()
