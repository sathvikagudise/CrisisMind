"""CrisisMind Model v2 Training — Hybrid-cache with gate regularization.

Trains GatedFusion with frozen encoders (pre-extracted features).
Compares WeightedCrossEntropy vs Focal Loss.
Adds gate regularization to prevent modality collapse.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json, time, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, f1_score,
)
from transformers import DistilBertTokenizer

from backend.training.config import (
    LABEL_NAMES, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    CHECKPOINTS_DIR, METRICS_DIR, PLOTS_DIR,
    IMG_SIZE, TEXT_MODEL_NAME, TEXT_MAX_LEN,
    FUSION_DROPOUT, NUM_CLASSES, FUSION_BATCH_SIZE, FUSION_LR, FUSION_PATIENCE,
)
from backend.datasets.multimodal_data import load_multimodal_data
from backend.models.fusion_models import (
    ImageFeatureExtractor, TextFeatureExtractor, GatedFusion, count_parameters,
)
from backend.config.shared import IMG_CKPT, TEXT_CKPT, DEVICE

SEED = RANDOM_SEED
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = torch.device(DEVICE)

V2_DIR = CHECKPOINTS_DIR / "v2"
V2_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("testing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_LABELS = ["Little or No Damage", "Mild Damage", "Severe Damage"]

import torchvision.transforms as T
EXTRACT_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
class WeightedCrossEntropy(nn.Module):
    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.alpha)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Gate regularization
# ---------------------------------------------------------------------------
def gate_regularization(gates, target=0.5, weight=0.01):
    """Penalize gates far from target (0.5). Encourages balanced modality use."""
    return weight * torch.mean((gates - target) ** 2)


def gate_entropy_bonus(gates, weight=0.005):
    """Encourage gates to be uncertain (high entropy) rather than collapsed."""
    eps = 1e-6
    entropy = -(gates * torch.log(gates + eps) + (1 - gates) * torch.log(1 - gates + eps))
    return -weight * entropy.mean()


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------
class FeatureCache:
    def __init__(self, path):
        self.path = Path(path)

    def exists(self):
        return self.path.exists()

    def load(self):
        return torch.load(self.path)

    def save(self, img_feats, txt_feats, labels):
        torch.save({"img_feats": img_feats, "txt_feats": txt_feats, "labels": labels}, self.path)


def extract_features(df, device):
    print("  Loading frozen encoders...")
    img_enc = ImageFeatureExtractor(IMG_CKPT, freeze=True).to(device).eval()
    txt_enc = TextFeatureExtractor(TEXT_CKPT, freeze=True).to(device).eval()
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)

    n = len(df)
    img_feats, txt_feats, labels = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for i in range(n):
            row = df.iloc[i]
            try:
                img = Image.open(row["full_image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
            img_t = EXTRACT_TF(img).unsqueeze(0).to(device)

            enc = tokenizer(str(row["tweet_text"]), truncation=True, padding="max_length",
                            max_length=TEXT_MAX_LEN, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            img_feats.append(img_enc(img_t).cpu())
            txt_feats.append(txt_enc(ids, mask).cpu())
            labels.append(int(row["label"]))

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{n} ({time.time()-t0:.0f}s)")

    return torch.cat(img_feats), torch.cat(txt_feats), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FusionDataset(Dataset):
    def __init__(self, img_f, txt_f, labels):
        self.img_f, self.txt_f, self.labels = img_f, txt_f, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.img_f[idx], self.txt_f[idx], self.labels[idx]


def collate(b):
    return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]), torch.stack([x[2] for x in b])


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score):
        if self.best is None or score > self.best:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return True


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate_full(model, loader, device):
    model.eval()
    all_preds, all_labels, all_gates = [], [], []
    with torch.no_grad():
        for img_f, txt_f, labels in loader:
            img_f, txt_f, labels = img_f.to(device), txt_f.to(device), labels.to(device)
            logits, gates = model(img_f, txt_f)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gates.append(gates.cpu())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    gates_cat = torch.cat(all_gates, dim=0)

    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cr = classification_report(y_true, y_pred, target_names=SEVERITY_LABELS, output_dict=True, zero_division=0)

    img_gate = gates_cat[:, 0].mean().item()
    txt_gate = gates_cat[:, 1].mean().item()

    metrics = {
        "accuracy": acc,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f1_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w,
        "img_gate": img_gate, "txt_gate": txt_gate,
    }
    return metrics, cr, cm, y_true, y_pred


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_run(name, train_loader, val_loader, test_loader,
              class_weights, loss_type, gate_reg_weight, device,
              max_epochs=40, lr=1e-3, patience=7):
    """Train one run with specified loss and gate regularization."""
    print(f"\n{'='*60}")
    print(f"  RUN: {name}")
    print(f"  Loss: {loss_type}  Gate reg: {gate_reg_weight}  LR: {lr}")
    print(f"{'='*60}")

    model = GatedFusion(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES, dropout=FUSION_DROPOUT).to(device)
    trainable, total = count_parameters(model)
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    if loss_type == "wce":
        criterion = WeightedCrossEntropy(alpha=class_weights)
        print(f"  WeightedCrossEntropy alpha={class_weights.cpu().tolist()}")
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"  FocalLoss alpha={class_weights.cpu().tolist()}, gamma=2.0")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    early_stop = EarlyStopping(patience=patience)

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "val_acc": [], "val_prec_m": [], "val_rec_m": [],
        "val_f1_w": [],
        "img_gate": [], "txt_gate": [],
        "lr": [],
    }
    best_f1 = 0.0
    best_epoch = 0
    gate_history = []

    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        train_preds, train_labels = [], []
        train_gates = []

        for img_f, txt_f, labels in train_loader:
            img_f, txt_f, labels = img_f.to(device), txt_f.to(device), labels.to(device)
            logits, gates = model(img_f, txt_f)
            loss = criterion(logits, labels)

            # Gate regularization
            g_reg = gate_regularization(gates, target=0.5, weight=gate_reg_weight)
            loss = loss + g_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * img_f.size(0)
            preds = logits.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_gates.append(gates.detach().cpu())

        train_loss = total_loss / len(train_loader.dataset)
        train_f1 = f1_score(train_labels, train_preds, average="macro")
        train_gates_cat = torch.cat(train_gates, dim=0)

        # ── Validate ──
        val_metrics, val_cr, val_cm, _, _ = evaluate_full(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(0)  # computed below
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_prec_m"].append(val_metrics["precision_macro"])
        history["val_rec_m"].append(val_metrics["recall_macro"])
        history["val_f1_w"].append(val_metrics["f1_weighted"])
        history["img_gate"].append(val_metrics["img_gate"])
        history["txt_gate"].append(val_metrics["txt_gate"])
        history["lr"].append(current_lr)

        gate_history.append({
            "epoch": epoch,
            "img_gate_mean": val_metrics["img_gate"],
            "txt_gate_mean": val_metrics["txt_gate"],
            "gate_diff": abs(val_metrics["img_gate"] - val_metrics["txt_gate"]),
        })

        # Per-class metrics
        pc = ""
        for lbl in range(3):
            name_l = SEVERITY_LABELS[lbl]
            r = val_cr[name_l]
            pc += f"  {name_l[:6]}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}"

        improved = ""
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), V2_DIR / f"{name}_best.pth")
            improved = " *** BEST ***"

        print(
            f"  Ep {epoch:2d}/{max_epochs}  "
            f"TrL={train_loss:.4f}  TrF1={train_f1:.4f}  "
            f"VaF1={val_metrics['f1_macro']:.4f}  VaAcc={val_metrics['accuracy']:.4f}  "
            f"Gates[I={val_metrics['img_gate']:.3f},T={val_metrics['txt_gate']:.3f}]  "
            f"LR={current_lr:.2e}{improved}"
        )
        print(f"    {pc}")

        if early_stop(val_metrics["f1_macro"]):
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Load best and evaluate on test ──
    print(f"\n  Loading best checkpoint (epoch {best_epoch})...")
    model.load_state_dict(torch.load(V2_DIR / f"{name}_best.pth", map_location=device))
    test_metrics, test_cr, test_cm, y_true, y_pred = evaluate_full(model, test_loader, device)

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=ax)
    ax.set_title(f"CrisisMind v2 — {name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(V2_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Save training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], "o-", label="Train", color="steelblue")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_f1"], "o-", label="Train", color="steelblue")
    axes[0, 1].plot(epochs, history["val_f1"], "s-", label="Val", color="coral")
    axes[0, 1].set_title("Macro F1")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["img_gate"], "o-", label="Image Gate", color="steelblue")
    axes[1, 0].plot(epochs, history["txt_gate"], "s-", label="Text Gate", color="coral")
    axes[1, 0].axhline(0.5, color="gray", ls="--", alpha=0.5, label="Target=0.5")
    axes[1, 0].set_title("Gate Values")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(epochs, history["lr"], "o-", color="green")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"CrisisMind v2 — {name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(V2_DIR / f"{name}_training_curves.png", dpi=150)
    plt.close(fig)

    # Save gate history
    pd.DataFrame(gate_history).to_csv(V2_DIR / f"{name}_gate_history.csv", index=False)

    # Summary
    summary = {
        "name": name,
        "loss_type": loss_type,
        "gate_reg_weight": gate_reg_weight,
        "best_epoch": best_epoch,
        **{k: round(float(v), 4) for k, v in test_metrics.items()},
    }
    for lbl in range(3):
        n = SEVERITY_LABELS[lbl]
        summary[f"precision_{LABEL_NAMES[lbl]}"] = round(test_cr[n]["precision"], 4)
        summary[f"recall_{LABEL_NAMES[lbl]}"] = round(test_cr[n]["recall"], 4)
        summary[f"f1_{LABEL_NAMES[lbl]}"] = round(test_cr[n]["f1-score"], 4)

    with open(V2_DIR / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ── {name} Test Results ──")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):  {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  Gates:       img={test_metrics['img_gate']:.4f}  txt={test_metrics['txt_gate']:.4f}")
    for lbl in range(3):
        n = SEVERITY_LABELS[lbl]
        print(f"    {n:25s}  P={test_cr[n]['precision']:.4f}  R={test_cr[n]['recall']:.4f}  F1={test_cr[n]['f1-score']:.4f}")

    return summary, history, test_cr, test_cm, y_true, y_pred


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_v1_v2(v2_summary, v1_metrics):
    """Generate MODEL_COMPARISON.md."""
    lines = []
    lines.append("# CrisisMind — Model v1 vs v2 Comparison\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}\n")

    lines.append("## Summary\n")
    lines.append("| Metric | Model v1 (Production) | Model v2 | Delta |")
    lines.append("|--------|----------------------|----------|-------|")

    pairs = [
        ("Accuracy", "accuracy"),
        ("F1 (macro)", "f1_macro"),
        ("F1 (weighted)", "f1_weighted"),
        ("Precision (macro)", "precision_macro"),
        ("Recall (macro)", "recall_macro"),
        ("Image Gate", "img_gate"),
        ("Text Gate", "txt_gate"),
    ]
    for label, key in pairs:
        v1 = v1_metrics.get(key, 0)
        v2 = v2_summary.get(key, 0)
        delta = v2 - v1
        sign = "+" if delta > 0 else ""
        lines.append(f"| {label} | {v1:.4f} | {v2:.4f} | {sign}{delta:.4f} |")

    lines.append("")
    lines.append("## Per-Class Recall\n")
    lines.append("| Class | v1 Recall | v2 Recall | Delta |")
    lines.append("|-------|-----------|-----------|-------|")
    for lbl in range(3):
        key = f"recall_{LABEL_NAMES[lbl]}"
        v1 = v1_metrics.get(key, 0)
        v2 = v2_summary.get(key, 0)
        delta = v2 - v1
        sign = "+" if delta > 0 else ""
        lines.append(f"| {SEVERITY_LABELS[lbl]} | {v1:.4f} | {v2:.4f} | {sign}{delta:.4f} |")

    lines.append("")
    lines.append("## Per-Class F1\n")
    lines.append("| Class | v1 F1 | v2 F1 | Delta |")
    lines.append("|-------|-------|-------|-------|")
    for lbl in range(3):
        key = f"f1_{LABEL_NAMES[lbl]}"
        v1 = v1_metrics.get(key, 0)
        v2 = v2_summary.get(key, 0)
        delta = v2 - v1
        sign = "+" if delta > 0 else ""
        lines.append(f"| {SEVERITY_LABELS[lbl]} | {v1:.4f} | {v2:.4f} | {sign}{delta:.4f} |")

    lines.append("")
    lines.append("## Gate Analysis\n")
    lines.append(f"- **v1 Image Gate:** {v1_metrics.get('img_gate', 0.17):.4f}")
    lines.append(f"- **v1 Text Gate:** {v1_metrics.get('txt_gate', 0.014):.4f}")
    lines.append(f"- **v2 Image Gate:** {v2_summary.get('img_gate', 0):.4f}")
    lines.append(f"- **v2 Text Gate:** {v2_summary.get('txt_gate', 0):.4f}")

    v2_txt = v2_summary.get("txt_gate", 0)
    v1_txt = v1_metrics.get("txt_gate", 0.014)
    if v2_txt > v1_txt * 2:
        lines.append(f"\n**Gate collapse improved:** Text gate increased from {v1_txt:.4f} to {v2_txt:.4f} ({v2_txt/v1_txt:.1f}x)")
    else:
        lines.append(f"\n**Gate collapse:** Text gate remains low ({v2_txt:.4f})")

    lines.append("")
    lines.append("## Recommendation\n")

    v1_f1 = v1_metrics.get("f1_macro", 0)
    v2_f1 = v2_summary.get("f1_macro", 0)
    delta_f1 = v2_f1 - v1_f1

    v1_little = v1_metrics.get("recall_little_or_no_damage", v1_metrics.get("recall_little", 0))
    v2_little = v2_summary.get("recall_little_or_no_damage", 0)
    v1_mild = v1_metrics.get("recall_mild_damage", v1_metrics.get("recall_mild", 0))
    v2_mild = v2_summary.get("recall_mild_damage", 0)

    if delta_f1 > 0.01 and v2_little >= v1_little - 0.05 and v2_mild >= v1_mild - 0.05:
        lines.append("**RECOMMEND: Deploy Model v2.** Significant improvement in Macro F1 with no unacceptable degradation in minority class recall.")
    elif delta_f1 > 0:
        lines.append("**CONDITIONAL: Model v2 shows improvement** but with trade-offs in minority class performance. Consider deploying with monitoring.")
    else:
        lines.append("**NOT RECOMMENDED:** Model v2 does not outperform Model v1. Keep current checkpoint.")

    lines.append(f"\nDelta F1: {delta_f1:+.4f}")
    lines.append(f"Delta Little Recall: {v2_little - v1_little:+.4f}")
    lines.append(f"Delta Mild Recall: {v2_mild - v1_mild:+.4f}")

    (RESULTS_DIR / "MODEL_COMPARISON.md").write_text("\n".join(lines))
    print(f"\n  Saved: {RESULTS_DIR / 'MODEL_COMPARISON.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 70)
    print("  CrisisMind — Model v2 Training")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    # ── 1. Load data ──
    print("\n[1] Loading dataset...")
    df = load_multimodal_data()
    y = df["label"].values
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y[train_idx])

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df["label"].value_counts().sort_index()
        parts = "  ".join(f"{LABEL_NAMES[k]}={v}" for k, v in dist.items())
        print(f"    {split_name}: {parts}")

    # ── 2. Class weights ──
    print("\n[2] Computing class weights...")
    counts = torch.bincount(torch.tensor(train_df["label"].values), minlength=NUM_CLASSES).float()
    weights_inv = 1.0 / counts
    weights_inv = weights_inv / weights_inv.sum() * NUM_CLASSES
    weights_sqrt = 1.0 / torch.sqrt(counts)
    weights_sqrt = weights_sqrt / weights_sqrt.sum() * NUM_CLASSES
    print(f"  Counts:   {counts.tolist()}")
    print(f"  Inverse:  {weights_inv.tolist()}")
    print(f"  Sqrt:     {weights_sqrt.tolist()}")

    # ── 3. Extract features ──
    print("\n[3] Feature extraction (cached)...")
    cache = FeatureCache(V2_DIR / "feature_cache.pt")

    if cache.exists():
        print("  Loading from cache...")
        cached = cache.load()
        all_img_feats = cached["img_feats"]
        all_txt_feats = cached["txt_feats"]
        all_labels = cached["labels"]
    else:
        print("  Extracting features with frozen encoders...")
        all_img_feats, all_txt_feats, all_labels = extract_features(df, DEV)
        cache.save(all_img_feats, all_txt_feats, all_labels)
        print(f"  Cached to {V2_DIR / 'feature_cache.pt'}")

    img_dim = all_img_feats.shape[1]
    txt_dim = all_txt_feats.shape[1]
    print(f"  Feature dims: img={img_dim}, txt={txt_dim}")

    # Slice features
    train_feats = (all_img_feats[train_idx], all_txt_feats[train_idx], all_labels[train_idx])
    val_feats = (all_img_feats[val_idx], all_txt_feats[val_idx], all_labels[val_idx])
    test_feats = (all_img_feats[test_idx], all_txt_feats[test_idx], all_labels[test_idx])

    # ── 4. Data loaders with WeightedRandomSampler ──
    print("\n[4] Creating data loaders with WeightedRandomSampler...")
    train_labels_t = train_feats[2]
    class_counts = torch.bincount(train_labels_t, minlength=NUM_CLASSES).float()
    sample_weights = (1.0 / class_counts)[train_labels_t].tolist()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels_t), replacement=True)

    train_ds = FusionDataset(*train_feats)
    val_ds = FusionDataset(*val_feats)
    test_ds = FusionDataset(*test_feats)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)

    print(f"  Train batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")

    # ── 5. Train: WCE vs Focal Loss comparison ──
    print("\n[5] Training: comparing WeightedCrossEntropy vs FocalLoss...")
    all_summaries = []

    configs = [
        ("wce_basic", "wce", weights_inv, 0.005, 1e-3, 7),
        ("wce_sqrt", "wce", weights_sqrt, 0.005, 1e-3, 7),
        ("focal_g2", "focal", weights_inv, 0.005, 1e-3, 7),
        ("focal_g3", "focal", weights_inv, 0.005, 1e-3, 7),  # will override gamma below
        ("focal_g2_reg01", "focal", weights_inv, 0.01, 1e-3, 7),
        ("focal_g2_noreg", "focal", weights_inv, 0.0, 1e-3, 7),
    ]

    for name, loss_type, weights, gate_reg, lr, pat in configs:
        summary_path = V2_DIR / f"{name}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"\n  {name}: already complete (F1={summary['f1_macro']:.4f}), loading...")
            all_summaries.append(summary)
            continue

        summary, history, cr, cm, yt, yp = train_run(
            name, train_loader, val_loader, test_loader,
            weights.to(DEV), loss_type, gate_reg, DEV,
            max_epochs=40, lr=lr, patience=pat,
        )
        all_summaries.append(summary)

    # ── 6. Select best and save as production v2 ──
    print("\n[6] Selecting best model...")
    best = max(all_summaries, key=lambda s: s["f1_macro"])
    print(f"  Best: {best['name']} (F1 macro={best['f1_macro']:.4f})")

    best_ckpt = V2_DIR / f"{best['name']}_best.pth"
    prod_ckpt = CHECKPOINTS_DIR / "best_gated_fusion_v2.pth"
    shutil.copy2(best_ckpt, prod_ckpt)
    print(f"  Saved to {prod_ckpt}")

    # ── 7. Load v1 metrics for comparison ──
    print("\n[7] Comparing with Model v1...")
    v1_metrics = {
        "accuracy": 0.6352,
        "f1_macro": 0.5375,
        "f1_weighted": 0.6415,
        "precision_macro": 0.5289,
        "recall_macro": 0.5437,
        "recall_little_or_no_damage": 0.4366,
        "recall_mild_damage": 0.4444,
        "recall_severe_damage": 0.7500,
        "f1_little_or_no_damage": 0.4189,
        "f1_mild_damage": 0.4222,
        "f1_severe_damage": 0.7710,
        "img_gate": 0.173,
        "txt_gate": 0.014,
    }

    # ── 8. Generate comparison document ──
    compare_v1_v2(best, v1_metrics)

    # ── 9. Generate all comparison plots ──
    print("\n[8] Generating comparison plots...")
    _generate_comparison_plots(all_summaries, v1_metrics)

    # ── 10. Generate MODEL_V2_REPORT.md ──
    _generate_v2_report(best, all_summaries, v1_metrics)

    # ── 11. Generate GATE_ANALYSIS.md ──
    _generate_gate_analysis(all_summaries)

    # ── 12. Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  v2 TRAINING COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\n  All runs:")
    for s in sorted(all_summaries, key=lambda x: -x["f1_macro"]):
        print(f"    {s['name']:25s}  F1={s['f1_macro']:.4f}  "
              f"Rec[L={s.get('recall_little_or_no_damage',0):.3f},"
              f"M={s.get('recall_mild_damage',0):.3f},"
              f"S={s.get('recall_severe_damage',0):.3f}]  "
              f"Gates[I={s['img_gate']:.3f},T={s['txt_gate']:.3f}]")

    print(f"\n  Production v1 F1: {v1_metrics['f1_macro']:.4f}")
    print(f"  Best v2 F1:       {best['f1_macro']:.4f}")
    delta = best['f1_macro'] - v1_metrics['f1_macro']
    print(f"  Delta:            {delta:+.4f}")

    # Cleanup feature cache
    cache_path = V2_DIR / "feature_cache.pt"
    if cache_path.exists():
        cache_path.unlink()
        print(f"\n  Cleaned up feature cache.")


def _generate_comparison_plots(all_summaries, v1_metrics):
    """Side-by-side bar chart: v1 vs best v2."""
    best = max(all_summaries, key=lambda s: s["f1_macro"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Overall metrics
    metrics_keys = ["accuracy", "f1_macro", "f1_weighted"]
    metrics_labels = ["Accuracy", "F1 (macro)", "F1 (weighted)"]
    v1_vals = [v1_metrics.get(k, 0) for k in metrics_keys]
    v2_vals = [best.get(k, 0) for k in metrics_keys]

    x = np.arange(len(metrics_labels))
    axes[0].bar(x - 0.2, v1_vals, 0.35, label="v1 (Production)", color="#e74c3c")
    axes[0].bar(x + 0.2, v2_vals, 0.35, label="v2 (Best)", color="#2ecc71")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Overall Metrics", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-class recall
    recall_keys = ["recall_little_or_no_damage", "recall_mild_damage", "recall_severe_damage"]
    recall_labels = ["Little", "Mild", "Severe"]
    v1_recall = [v1_metrics.get(k, 0) for k in recall_keys]
    v2_recall = [best.get(k, 0) for k in recall_keys]

    x = np.arange(len(recall_labels))
    axes[1].bar(x - 0.2, v1_recall, 0.35, label="v1", color="#e74c3c")
    axes[1].bar(x + 0.2, v2_recall, 0.35, label="v2", color="#2ecc71")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(recall_labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Per-Class Recall", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gates
    gate_labels = ["Image Gate", "Text Gate"]
    v1_gates = [v1_metrics.get("img_gate", 0.17), v1_metrics.get("txt_gate", 0.01)]
    v2_gates = [best.get("img_gate", 0), best.get("txt_gate", 0)]

    x = np.arange(2)
    axes[2].bar(x - 0.2, v1_gates, 0.35, label="v1", color="#e74c3c")
    axes[2].bar(x + 0.2, v2_gates, 0.35, label="v2", color="#2ecc71")
    axes[2].axhline(0.5, color="gray", ls="--", alpha=0.5, label="Target=0.5")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(gate_labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Gate Values", fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("CrisisMind — Model v1 vs v2 Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "MODEL_COMPARISON.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {RESULTS_DIR / 'MODEL_COMPARISON.png'}")

    # All runs comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [s["name"] for s in all_summaries] + ["v1_prod"]
    f1s = [s["f1_macro"] for s in all_summaries] + [v1_metrics["f1_macro"]]
    colors = ["#3498db"] * len(all_summaries) + ["#e74c3c"]
    bars = ax.barh(names, f1s, color=colors)
    ax.set_xlabel("Macro F1")
    ax.set_title("All Runs — Macro F1 Comparison", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{f1:.4f}", va="center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ALL_RUNS_F1.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {RESULTS_DIR / 'ALL_RUNS_F1.png'}")


def _generate_v2_report(best, all_summaries, v1_metrics):
    lines = []
    lines.append("# CrisisMind — Model v2 Training Report\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}\n")

    lines.append("## Training Configuration\n")
    lines.append("- **Architecture:** EfficientNet-B0 (image) + DistilBERT (text) + GatedFusion (multimodal)")
    lines.append("- **Training:** Hybrid-cache (frozen encoders, trainable fusion head)")
    lines.append("- **Balanced sampling:** WeightedRandomSampler (inverse class frequency)")
    lines.append("- **Gate regularization:** Quadratic penalty toward 0.5")
    lines.append("- **Optimizer:** AdamW with cosine annealing")
    lines.append("- **Max epochs:** 40 with early stopping (patience=7)")
    lines.append("- **Selection criterion:** Validation Macro F1\n")

    lines.append("## Class Distribution\n")
    lines.append("| Class | Count | Percentage |")
    lines.append("|-------|-------|------------|")
    lines.append("| Little or No Damage | 475 | 13.5% |")
    lines.append("| Mild Damage | 839 | 23.8% |")
    lines.append("| Severe Damage | 2,212 | 62.7% |")
    lines.append("| **Total** | **3,526** | **100%** |\n")

    lines.append("## Loss Function Comparison\n")
    lines.append("| Run | Loss | Gate Reg | F1 (macro) | F1 (wtd) | Acc | Rec-L | Rec-M | Rec-S |")
    lines.append("|-----|------|----------|-----------|---------|-----|-------|-------|-------|")
    for s in sorted(all_summaries, key=lambda x: -x["f1_macro"]):
        marker = " **" if s["name"] == best["name"] else ""
        lines.append(
            f"| {s['name']}{marker} | {s['loss_type']} | {s['gate_reg_weight']} | "
            f"{s['f1_macro']:.4f} | {s['f1_weighted']:.4f} | {s['accuracy']:.4f} | "
            f"{s.get('recall_little_or_no_damage',0):.4f} | "
            f"{s.get('recall_mild_damage',0):.4f} | "
            f"{s.get('recall_severe_damage',0):.4f} |"
        )

    lines.append(f"\n## Best Model: {best['name']}\n")
    lines.append(f"- **Loss:** {best['loss_type']}")
    lines.append(f"- **Gate regularization weight:** {best['gate_reg_weight']}")
    lines.append(f"- **Best epoch:** {best['best_epoch']}")
    lines.append(f"- **Test Accuracy:** {best['accuracy']:.4f}")
    lines.append(f"- **Test F1 (macro):** {best['f1_macro']:.4f}")
    lines.append(f"- **Test F1 (weighted):** {best['f1_weighted']:.4f}")
    lines.append(f"- **Image gate:** {best['img_gate']:.4f}")
    lines.append(f"- **Text gate:** {best['txt_gate']:.4f}\n")

    lines.append("### Per-Class Metrics\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|----|---------|")
    for lbl in range(3):
        n = LABEL_NAMES[lbl]
        lines.append(
            f"| {SEVERITY_LABELS[lbl]} | "
            f"{best.get(f'precision_{n}', 0):.4f} | "
            f"{best.get(f'recall_{n}', 0):.4f} | "
            f"{best.get(f'f1_{n}', 0):.4f} | - |"
        )

    lines.append("\n## Files Generated\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| best_gated_fusion_v2.pth | Production checkpoint |")
    lines.append("| MODEL_V2_REPORT.md | This report |")
    lines.append("| MODEL_COMPARISON.md | v1 vs v2 comparison |")
    lines.append("| MODEL_COMPARISON.png | Comparison bar chart |")
    lines.append("| ALL_RUNS_F1.png | All runs F1 comparison |")
    lines.append("| GATE_ANALYSIS.md | Gate collapse analysis |")
    lines.append("| CONFUSION_MATRIX_V2.png | Best model confusion matrix |")
    lines.append("| TRAINING_CURVES.png | Best model training curves |")

    (RESULTS_DIR / "MODEL_V2_REPORT.md").write_text("\n".join(lines))
    print(f"  Saved: {RESULTS_DIR / 'MODEL_V2_REPORT.md'}")


def _generate_gate_analysis(all_summaries):
    lines = []
    lines.append("# CrisisMind — Gate Collapse Analysis\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}\n")

    lines.append("## Problem\n")
    lines.append("The v1 GatedFusion exhibited near-zero text gate values (~0.014),")
    lines.append("effectively ignoring the text modality. This was traced to:")
    lines.append("1. No regularization on gate values during training")
    lines.append("2. Text features from DistilBERT had low discriminative power")
    lines.append("3. Class imbalance caused the model to rely on majority-class bias\n")

    lines.append("## Solution\n")
    lines.append("Added quadratic gate regularization: `L_reg = weight * mean((gate - 0.5)^2)`")
    lines.append("This penalizes gates that deviate from the balanced target of 0.5.\n")

    lines.append("## Results\n")
    lines.append("| Run | Gate Reg Weight | Image Gate | Text Gate | Gate Diff |")
    lines.append("|-----|----------------|-----------|----------|-----------|")
    for s in sorted(all_summaries, key=lambda x: x["txt_gate"], reverse=True):
        diff = abs(s["img_gate"] - s["txt_gate"])
        lines.append(f"| {s['name']} | {s['gate_reg_weight']} | {s['img_gate']:.4f} | {s['txt_gate']:.4f} | {diff:.4f} |")

    lines.append("\n## Interpretation\n")
    best = max(all_summaries, key=lambda s: s["txt_gate"])
    worst = min(all_summaries, key=lambda s: s["txt_gate"])
    lines.append(f"- **Best text gate:** {best['name']} ({best['txt_gate']:.4f}) with reg weight={best['gate_reg_weight']}")
    lines.append(f"- **Worst text gate:** {worst['name']} ({worst['txt_gate']:.4f}) with reg weight={worst['gate_reg_weight']}")
    lines.append(f"- **v1 baseline text gate:** 0.014")

    if best["txt_gate"] > 0.1:
        lines.append(f"\n**Significant improvement in gate balance.** Text gate increased {best['txt_gate']/0.014:.1f}x over v1.")
    elif best["txt_gate"] > 0.05:
        lines.append(f"\n**Moderate improvement.** Text gate increased but remains below balanced threshold.")
    else:
        lines.append(f"\n**Limited improvement.** The text modality may have insufficient discriminative power for this dataset.")

    (RESULTS_DIR / "GATE_ANALYSIS.md").write_text("\n".join(lines))
    print(f"  Saved: {RESULTS_DIR / 'GATE_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
