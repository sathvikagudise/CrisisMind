"""Hybrid-cache experiments: pre-extract features, train fusion heads, compare runs."""
import json
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
EXPERIMENTS_ROOT = Path("experiments")
DEV = torch.device(DEVICE)

# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
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
# Feature caching
# ---------------------------------------------------------------------------
def extract_and_cache_features(df, device):
    """Run frozen encoders once and return cached feature tensors + labels."""
    print("  Loading encoders...")
    img_encoder = ImageFeatureExtractor(IMG_CKPT, freeze=True).to(device)
    txt_encoder = TextFeatureExtractor(TEXT_CKPT, freeze=True).to(device)
    img_encoder.eval()
    txt_encoder.eval()

    img_dim = img_encoder.out_dim
    txt_dim = txt_encoder.out_dim
    n = len(df)
    print(f"  Extracting features for {n} samples (img_dim={img_dim}, txt_dim={txt_dim})...")

    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)

    all_img_feats = []
    all_txt_feats = []
    all_labels = []

    with torch.no_grad():
        for idx in range(n):
            row = df.iloc[idx]
            img_path = row["full_image_path"]
            text = str(row["tweet_text"])
            label = int(row["label"])

            try:
                pil_img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            except Exception:
                pil_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

            img_tensor = (
                torchvision_transforms(pil_img).unsqueeze(0).to(device)
            )
            encoded = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=TEXT_MAX_LEN, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)

            img_feat = img_encoder(img_tensor).cpu()
            txt_feat = txt_encoder(input_ids, attn_mask).cpu()
            all_img_feats.append(img_feat)
            all_txt_feats.append(txt_feat)
            all_labels.append(label)

            if (idx + 1) % 500 == 0:
                print(f"    Extracted {idx + 1}/{n}")

    img_feats = torch.cat(all_img_feats, dim=0)
    txt_feats = torch.cat(all_txt_feats, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"  Done. img_feats={img_feats.shape}, txt_feats={txt_feats.shape}, labels={labels.shape}")
    return img_feats, txt_feats, labels

# Standard image transform for feature extraction
import torchvision.transforms as torchvision_transforms
torchvision_transforms = torchvision_transforms.Compose([
    torchvision_transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision_transforms.ToTensor(),
    torchvision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Cached Dataset
# ---------------------------------------------------------------------------
class CachedFusionDataset(Dataset):
    """Loads pre-extracted features + labels."""
    def __init__(self, img_feats, txt_feats, labels):
        self.img_feats = img_feats
        self.txt_feats = txt_feats
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "img_feat": self.img_feats[idx],
            "txt_feat": self.txt_feats[idx],
            "label": self.labels[idx],
        }

def collate_cached(batch):
    img_feats = torch.stack([b["img_feat"] for b in batch])
    txt_feats = torch.stack([b["txt_feat"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return img_feats, txt_feats, labels

# ---------------------------------------------------------------------------
# Gate Monitor
# ---------------------------------------------------------------------------
class GateMonitor:
    def __init__(self):
        self.epochs = []
        self.img_gates = []
        self.txt_gates = []

    def record(self, epoch, gates):
        self.epochs.append(epoch)
        self.img_gates.append(gates[:, 0].mean().item())
        self.txt_gates.append(gates[:, 1].mean().item())

    def to_df(self):
        return pd.DataFrame({
            "epoch": self.epochs,
            "img_gate_mean": self.img_gates,
            "txt_gate_mean": self.txt_gates,
        })

    def save_plot(self, path):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.epochs, self.img_gates, "o-", label="Image Gate", color="steelblue")
        ax.plot(self.epochs, self.txt_gates, "s-", label="Text Gate", color="coral")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gate Value")
        ax.set_title("Gate Evolution", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"  ES: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return True
        self.best_score = score
        self.counter = 0
        return False

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_gates = [], [], []

    for img_feat, txt_feat, labels in loader:
        img_feat, txt_feat, labels = img_feat.to(device), txt_feat.to(device), labels.to(device)
        logits, gates = model(img_feat, txt_feat)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img_feat.size(0)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_gates.append(gates.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, torch.cat(all_gates, dim=0)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_gates = [], [], []

    with torch.no_grad():
        for img_feat, txt_feat, labels in loader:
            img_feat, txt_feat, labels = img_feat.to(device), txt_feat.to(device), labels.to(device)
            logits, gates = model(img_feat, txt_feat)
            loss = criterion(logits, labels)
            total_loss += loss.item() * img_feat.size(0)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gates.append(gates.cpu())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, torch.cat(all_gates, dim=0), all_labels, all_preds

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img_feat, txt_feat, labels in loader:
            img_feat, txt_feat, labels = img_feat.to(device), txt_feat.to(device), labels.to(device)
            logits, _ = model(img_feat, txt_feat)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    cr = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    metrics = {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
    }
    return metrics, cr, cm, y_true, y_pred

def compute_class_weights_tensor(train_labels, device, mode="inverse"):
    """mode: 'inverse' (1/count), 'sqrt' (1/sqrt(count)), 'effective' (1/log(1+count))."""
    counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float() + 1.0
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt":
        weights = 1.0 / torch.sqrt(counts)
    elif mode == "effective":
        weights = 1.0 / torch.log(1.0 + counts)
    else:
        weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return weights.to(device)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_training_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="coral")
    axes[0].set_title("Loss Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["train_f1"], "o-", label="Train F1", color="steelblue")
    axes[1].plot(epochs, history["val_f1"], "s-", label="Val F1", color="coral")
    axes[1].set_title("Macro F1 Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_confusion_matrix(cm, path, labels):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_classification_report_csv(cr, path):
    rows = []
    for label_key in sorted(LABEL_NAMES):
        name = LABEL_NAMES[label_key]
        r = cr[name]
        rows.append({"class": name, "precision": round(r["precision"], 4),
                      "recall": round(r["recall"], 4), "f1-score": round(r["f1-score"], 4),
                      "support": int(r["support"])})
    for avg in ["macro avg", "weighted avg"]:
        r = cr[avg]
        rows.append({"class": avg, "precision": round(r["precision"], 4),
                      "recall": round(r["recall"], 4), "f1-score": round(r["f1-score"], 4),
                      "support": int(r["support"])})
    pd.DataFrame(rows).to_csv(path, index=False)

# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------
def run_experiment(run_dir, config, train_feats, val_feats, test_feats,
                   train_labels, val_labels, test_labels, device):
    print(f"\n{'='*60}")
    print(f"  RUN: {run_dir.name}")
    print(f"  Config: {json.dumps(config, indent=2)}")
    print(f"{'='*60}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "hyperparams.json", "w") as f:
        json.dump(config, f, indent=2)

    # Datasets
    train_ds = CachedFusionDataset(train_feats["img"], train_feats["txt"], train_labels)
    val_ds = CachedFusionDataset(val_feats["img"], val_feats["txt"], val_labels)
    test_ds = CachedFusionDataset(test_feats["img"], test_feats["txt"], test_labels)

    # WeightedRandomSampler
    class_counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    sample_weights = (1.0 / class_counts)[train_labels].tolist()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

    bs = config.get("batch_size", FUSION_BATCH_SIZE)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, collate_fn=collate_cached)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_cached)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_cached)

    # Model
    img_dim = train_feats["img"].shape[1]
    txt_dim = train_feats["txt"].shape[1]
    dropout = config.get("dropout", FUSION_DROPOUT)
    model = GatedFusion(img_dim=img_dim, txt_dim=txt_dim, num_classes=NUM_CLASSES, dropout=dropout).to(device)
    trainable, total = count_parameters(model)
    print(f"  Model: {trainable:,} trainable / {total:,} total")

    # Loss
    weight_mode = config.get("weight_mode", "inverse")
    alpha = compute_class_weights_tensor(train_labels, device, mode=weight_mode)
    gamma = config.get("gamma", 2.0)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    print(f"  Loss: FocalLoss alpha={alpha.cpu().tolist()}, gamma={gamma}")

    # Optimizer
    lr = config.get("lr", FUSION_LR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, verbose=False,
    )

    patience = config.get("patience", FUSION_PATIENCE)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    max_epochs = config.get("max_epochs", 30)

    # Gate monitor
    gate_monitor = GateMonitor()

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_f1, train_gates = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_gates, val_labels_np, val_preds_np = validate_one_epoch(
            model, val_loader, criterion, device,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        gate_monitor.record(epoch, val_gates)
        avg_img_gate = val_gates[:, 0].mean().item()
        avg_txt_gate = val_gates[:, 1].mean().item()

        print(
            f"  Epoch {epoch:2d}/{max_epochs}  "
            f"Train Loss: {train_loss:.4f}  Train F1: {train_f1:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val F1: {val_f1:.4f}  "
            f"Gates [img={avg_img_gate:.4f}, txt={avg_txt_gate:.4f}]  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "checkpoint.pth")
            print(f"  --> Best so far (Val Macro F1: {val_f1:.4f})")

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Save outputs
    # Training curves
    plot_training_curves(history, run_dir / "training_curves.png")

    # Gate data
    gate_monitor.to_df().to_csv(run_dir / "gate_statistics.csv", index=False)
    gate_monitor.save_plot(run_dir / "gate_evolution.png")

    # Load best checkpoint and evaluate on test set
    best_ckpt = run_dir / "checkpoint.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    metrics, cr, cm, y_true, y_pred = evaluate(model, test_loader, device)

    # Confusion matrix
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    plot_confusion_matrix(cm, run_dir / "confusion_matrix.png", target_names)

    # Classification report CSV
    save_classification_report_csv(cr, run_dir / "classification_report.csv")

    # Summary metrics
    summary = {
        "run": run_dir.name,
        "config": config,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        **{k: round(float(v), 4) for k, v in metrics.items()},
    }
    # Per-class recall
    for label_key in sorted(LABEL_NAMES):
        name = LABEL_NAMES[label_key]
        summary[f"recall_{name}"] = round(cr[name]["recall"], 4)
        summary[f"f1_{name}"] = round(cr[name]["f1-score"], 4)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print per-class metrics
    print(f"\n  --- Test Results ---")
    print(f"  Test Macro F1: {metrics['f1_macro']:.4f}  Weighted F1: {metrics['f1_weighted']:.4f}  Acc: {metrics['accuracy']:.4f}")
    for label_key in sorted(LABEL_NAMES):
        name = LABEL_NAMES[label_key]
        r = cr[name]
        print(f"    {name:30s}  recall={r['recall']:.4f}  f1={r['f1-score']:.4f}  support={int(r['support'])}")

    return summary

# ---------------------------------------------------------------------------
# Compare all runs and select best
# ---------------------------------------------------------------------------
def compare_and_select(experiments_root, summaries, production_metrics):
    """Compare all runs, select best by Macro F1, save comparison table."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: All Runs vs Production")
    print(f"{'='*60}")

    # Build comparison table
    rows = []
    for s in summaries:
        rows.append({
            "run": s["run"],
            "accuracy": s["accuracy"],
            "f1_macro": s["f1_macro"],
            "f1_weighted": s["f1_weighted"],
            "recall_little": s.get("recall_little_or_no_damage", 0),
            "recall_mild": s.get("recall_mild_damage", 0),
            "recall_severe": s.get("recall_severe_damage", 0),
            "f1_little": s.get("f1_little_or_no_damage", 0),
            "f1_mild": s.get("f1_mild_damage", 0),
            "f1_severe": s.get("f1_severe_damage", 0),
        })

    # Add production model as baseline
    rows.append({
        "run": "production (current)",
        "accuracy": production_metrics.get("accuracy", 0),
        "f1_macro": production_metrics.get("f1_macro", 0),
        "f1_weighted": production_metrics.get("f1_weighted", 0),
        "recall_little": production_metrics.get("recall_little", 0),
        "recall_mild": production_metrics.get("recall_mild", 0),
        "recall_severe": production_metrics.get("recall_severe", 0),
        "f1_little": production_metrics.get("f1_little", 0),
        "f1_mild": production_metrics.get("f1_mild", 0),
        "f1_severe": production_metrics.get("f1_severe", 0),
    })

    comparison = pd.DataFrame(rows)
    comparison_path = experiments_root / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n{comparison.to_string(index=False)}")
    print(f"\n  Comparison saved: {comparison_path}")

    # Select best (excluding production, excluding incomplete runs)
    candidate_rows = [
        r for r in rows
        if r["run"] != "production (current)" and r["f1_macro"] > 0
    ]
    if not candidate_rows:
        print("  No experiment runs to compare.")
        return None

    best = max(candidate_rows, key=lambda r: r["f1_macro"])
    print(f"\n  Best run by Macro F1: {best['run']} (f1_macro={best['f1_macro']:.4f})")

    prod_row = [r for r in rows if r["run"] == "production (current)"][0]
    print(f"  Production: f1_macro={prod_row['f1_macro']:.4f}")

    # Check if best beats production
    beats_prod = best["f1_macro"] > prod_row["f1_macro"]
    recall_ok = True
    if beats_prod:
        # Check minority recall doesn't drop significantly
        for cls in ["recall_little", "recall_mild"]:
            if best[cls] < prod_row[cls] - 0.05:
                recall_ok = False
                print(f"  Warning: {cls} dropped from {prod_row[cls]:.4f} to {best[cls]:.4f}")

    if beats_prod and recall_ok:
        print(f"\n  *** Best run ({best['run']}) BEATS production. Deploying to best_run/ ***")
        src_run = experiments_root / best["run"]
        dst = experiments_root / "best_run"
        import shutil
        for f in ["checkpoint.pth", "hyperparams.json", "summary.json", "training_curves.png",
                  "confusion_matrix.png", "classification_report.csv", "gate_statistics.csv", "gate_evolution.png"]:
            src_file = src_run / f
            if src_file.exists():
                shutil.copy2(src_file, dst / f)
        print(f"  Deployed to {dst}")
        return best["run"]
    else:
        print(f"  Best run does NOT beat production (or recall regression). Keeping current checkpoint.")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  CrisisMind — Hybrid-Cache Fusion Experiments")
    print("=" * 70)
    device = DEV
    print(f"\nDevice: {device}")

    # ── 1. Load data ──
    print("\n[1] Loading multimodal dataset...")
    df = load_multimodal_data()
    print(f"  Total samples: {len(df)}")
    print(f"  Label distribution:")
    for lbl, count in df["label"].value_counts().sort_index().items():
        print(f"    {LABEL_NAMES[lbl]:30s}  {count}")

    # ── 2. Train/val/test split ──
    y = df["label"].values
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y[train_idx])
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ── 3. Pre-extract features ──
    print("\n[2] Pre-extracting features (one-time cache)...")
    cache_path = EXPERIMENTS_ROOT / "feature_cache.pt"
    if cache_path.exists():
        print("  Loading cached features from disk...")
        cache = torch.load(cache_path)
        img_feats_all = cache["img_feats"]
        txt_feats_all = cache["txt_feats"]
        labels_all = cache["labels"]
    else:
        img_feats_all, txt_feats_all, labels_all = extract_and_cache_features(df, device)
        torch.save({"img_feats": img_feats_all, "txt_feats": txt_feats_all, "labels": labels_all}, cache_path)
        print(f"  Cached to {cache_path}")

    # Slice by split
    train_feats = {
        "img": img_feats_all[train_idx],
        "txt": txt_feats_all[train_idx],
    }
    val_feats = {
        "img": img_feats_all[val_idx],
        "txt": txt_feats_all[val_idx],
    }
    test_feats = {
        "img": img_feats_all[test_idx],
        "txt": txt_feats_all[test_idx],
    }
    train_labels = labels_all[train_idx]
    val_labels = labels_all[val_idx]
    test_labels = labels_all[test_idx]

    print(f"  Train features: img={train_feats['img'].shape}, txt={train_feats['txt'].shape}")
    print(f"  Val features:   img={val_feats['img'].shape}, txt={val_feats['txt'].shape}")
    print(f"  Test features:  img={test_feats['img'].shape}, txt={test_feats['txt'].shape}")

    # ── 4. Define experiment configurations ──
    print("\n[3] Defining experiment configurations...")
    experiments = [
        {
            "name": "run_001",
            "config": {
                "lr": 1e-3,
                "gamma": 2.0,
                "weight_mode": "inverse",
                "batch_size": FUSION_BATCH_SIZE,
                "dropout": FUSION_DROPOUT,
                "patience": FUSION_PATIENCE,
                "max_epochs": 30,
                "description": "Baseline LR=1e-3, gamma=2.0, inverse weighting",
            },
        },
        {
            "name": "run_002",
            "config": {
                "lr": 5e-4,
                "gamma": 2.0,
                "weight_mode": "inverse",
                "batch_size": FUSION_BATCH_SIZE,
                "dropout": FUSION_DROPOUT,
                "patience": FUSION_PATIENCE,
                "max_epochs": 30,
                "description": "Lower LR=5e-4, gamma=2.0, inverse weighting",
            },
        },
        {
            "name": "run_003",
            "config": {
                "lr": 1e-3,
                "gamma": 3.0,
                "weight_mode": "inverse",
                "batch_size": FUSION_BATCH_SIZE,
                "dropout": FUSION_DROPOUT,
                "patience": FUSION_PATIENCE,
                "max_epochs": 30,
                "description": "Higher gamma=3.0, inverse weighting",
            },
        },
        {
            "name": "run_004",
            "config": {
                "lr": 1e-3,
                "gamma": 2.0,
                "weight_mode": "sqrt",
                "batch_size": FUSION_BATCH_SIZE,
                "dropout": FUSION_DROPOUT,
                "patience": FUSION_PATIENCE,
                "max_epochs": 30,
                "description": "LR=1e-3, gamma=2.0, sqrt weighting (softer)",
            },
        },
        {
            "name": "run_005",
            "config": {
                "lr": 1e-3,
                "gamma": 1.5,
                "weight_mode": "inverse",
                "batch_size": FUSION_BATCH_SIZE,
                "dropout": FUSION_DROPOUT,
                "patience": FUSION_PATIENCE,
                "max_epochs": 30,
                "description": "LR=1e-3, gamma=1.5, inverse weighting (milder focal)",
            },
        },
    ]

    # ── 5. Run experiments ──
    print(f"\n[4] Running {len(experiments)} experiments (skipping completed)...")
    summaries = []
    for exp in experiments:
        run_dir = EXPERIMENTS_ROOT / exp["name"]
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"  {exp['name']}: already complete (f1_macro={summary.get('f1_macro', '?'):.4f}), skipping")
            summaries.append(summary)
            continue
        summary = run_experiment(
            run_dir, exp["config"],
            train_feats, val_feats, test_feats,
            train_labels, val_labels, test_labels,
            device,
        )
        summaries.append(summary)

    # ── 6. Load production model metrics ──
    print("\n[5] Loading production model metrics for comparison...")
    # From the audit report: GatedFusion macro F1 = 0.5375
    production_metrics = {
        "accuracy": 0.6352,
        "f1_macro": 0.5375,
        "f1_weighted": 0.6415,
        "recall_little": 0.0,
        "recall_mild": 0.4211,
        "recall_severe": 0.7676,
        "f1_little": 0.0,
        "f1_mild": 0.4348,
        "f1_severe": 0.8387,
    }
    print(f"  Production GatedFusion: Acc={production_metrics['accuracy']:.4f}, "
          f"Macro F1={production_metrics['f1_macro']:.4f}")

    # ── 7. Compare and select best ──
    print("\n[6] Comparing all runs...")
    best_run = compare_and_select(EXPERIMENTS_ROOT, summaries, production_metrics)

    # ── 8. Final report ──
    print(f"\n{'='*70}")
    print(f"  EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Runs completed: {len(experiments)}")
    for s in summaries:
        print(f"    {s['run']:10s}  f1_macro={s['f1_macro']:.4f}  "
              f"recall_little={s.get('recall_little_or_no_damage', 0):.4f}  "
              f"recall_mild={s.get('recall_mild_damage', 0):.4f}  "
              f"recall_severe={s.get('recall_severe_damage', 0):.4f}")
    print(f"    {'production':10s}  f1_macro={production_metrics['f1_macro']:.4f}")
    if best_run:
        print(f"\n  Best run deployed to experiments/best_run/")
    else:
        print(f"\n  No improvement over production. Keeping current checkpoint.")

    # ── 9. Cleanup: remove feature cache to save disk space ──
    if cache_path.exists():
        cache_path.unlink()
        print(f"\n  Cleaned up feature cache.")

if __name__ == "__main__":
    main()
