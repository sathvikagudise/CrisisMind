"""Improved fusion training with balanced sampling, focal loss, staged fine-tuning, and gate monitoring."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from backend.training.config import (
    LABEL_NAMES, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    CHECKPOINTS_DIR, METRICS_DIR, PLOTS_DIR,
    IMG_SIZE, TEXT_MODEL_NAME, TEXT_MAX_LEN,
    FUSION_DROPOUT, NUM_CLASSES, FUSION_BATCH_SIZE, FUSION_LR, FUSION_PATIENCE,
)
from backend.datasets.multimodal_data import load_multimodal_data
from backend.models.fusion_models import (
    ImageFeatureExtractor, TextFeatureExtractor,
    GatedFusion, count_parameters,
)
from backend.config.shared import IMG_CKPT, TEXT_CKPT, DEVICE

SEED = RANDOM_SEED
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Augmentations: stronger for minority classes, minimal for majority
# ---------------------------------------------------------------------------
STRONG_AUG = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MINIMAL_AUG = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_AUG = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Dataset with class-aware augmentations
# ---------------------------------------------------------------------------
class ImprovedMultimodalDataset(Dataset):
    """Applies stronger augmentation to Little/Mild, minimal to Severe."""

    def __init__(self, dataframe, is_train=True):
        self.df = dataframe.reset_index(drop=True)
        self.is_train = is_train
        self.tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["full_image_path"]
        text = str(row["tweet_text"])
        label = int(row["label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        if self.is_train:
            # Stronger aug for minority classes (0=little, 1=mild)
            if label in (0, 1):
                image = STRONG_AUG(image)
            else:
                image = MINIMAL_AUG(image)
        else:
            image = VAL_AUG(image)

        encoded = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=TEXT_MAX_LEN, return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return images, input_ids, attention_mask, labels

# ---------------------------------------------------------------------------
# Focal Loss with class weights
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ---------------------------------------------------------------------------
# Gate monitor
# ---------------------------------------------------------------------------
class GateMonitor:
    def __init__(self):
        self.epochs = []
        self.img_gates = []
        self.txt_gates = []
        self.img_means = []
        self.txt_means = []

    def record(self, epoch, gates):
        """gates: tensor of shape (batch, 2) from validation."""
        img = gates[:, 0].mean().item()
        txt = gates[:, 1].mean().item()
        self.epochs.append(epoch)
        self.img_gates.append(img)
        self.txt_gates.append(txt)

    def save_csv(self, path):
        df = pd.DataFrame({
            "epoch": self.epochs,
            "img_gate_mean": self.img_gates,
            "txt_gate_mean": self.txt_gates,
        })
        df.to_csv(path, index=False)

    def save_plot(self, path):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.epochs, self.img_gates, "o-", label="Image Gate", color="steelblue")
        ax.plot(self.epochs, self.txt_gates, "s-", label="Text Gate", color="coral")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gate Value")
        ax.set_title("Gate Evolution During Training", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

# ---------------------------------------------------------------------------
# Training helpers
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
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return True
        self.best_score = score
        self.counter = 0
        return False

def compute_class_weights(train_df):
    counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Epoch runners
# ---------------------------------------------------------------------------
def train_epoch(model, img_encoder, txt_encoder, loader, criterion, optimizer, device):
    model.train()
    img_encoder.train()
    txt_encoder.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_gates = []

    for images, input_ids, attention_mask, labels in loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        img_feat = img_encoder(images)
        txt_feat = txt_encoder(input_ids, attention_mask)
        logits, gates = model(img_feat, txt_feat)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_gates.append(gates.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    gates_cat = torch.cat(all_gates, dim=0)
    return avg_loss, f1, gates_cat

def validate_epoch(model, img_encoder, txt_encoder, loader, criterion, device):
    model.eval()
    img_encoder.eval()
    txt_encoder.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_gates = []

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            img_feat = img_encoder(images)
            txt_feat = txt_encoder(input_ids, attention_mask)
            logits, gates = model(img_feat, txt_feat)

            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gates.append(gates.cpu())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    gates_cat = torch.cat(all_gates, dim=0)
    return avg_loss, f1, gates_cat, all_labels, all_preds

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, img_encoder, txt_encoder, loader, device):
    model.eval()
    img_encoder.eval()
    txt_encoder.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            img_feat = img_encoder(images)
            txt_feat = txt_encoder(input_ids, attention_mask)
            logits, gates = model(img_feat, txt_feat)

            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    clf_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return metrics, clf_report, cm, y_true, y_pred

def print_sanity_results(model, img_encoder, txt_encoder, test_df, device, n=10):
    """Run n samples per class and print predictions."""
    print("\n  --- Sanity Checks ---")
    all_rows = []
    for label in [0, 1, 2]:
        subset = test_df[test_df["label"] == label].head(n)
        for _, row in subset.iterrows():
            img_path = row["full_image_path"]
            text = str(row["tweet_text"])
            true_label = int(row["label"])

            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img_tensor = VAL_AUG(img).unsqueeze(0).to(device)

            tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
            encoded = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=TEXT_MAX_LEN, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                img_feat = img_encoder(img_tensor)
                txt_feat = txt_encoder(input_ids, attention_mask)
                logits, gates = model(img_feat, txt_feat)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0, pred_idx].item() * 100)

            all_rows.append({
                "true_label": LABEL_NAMES[true_label],
                "predicted": LABEL_NAMES[pred_idx],
                "confidence_pct": round(confidence, 2),
                "correct": "YES" if pred_idx == true_label else "NO",
                "text": text[:80],
            })

    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    return df

def plot_training_history(history, filename_prefix="training_v2"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="coral")
    axes[0].set_title("Loss Curves (Improved)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["train_f1"], "o-", label="Train F1", color="steelblue")
    axes[1].plot(epochs, history["val_f1"], "s-", label="Val F1", color="coral")
    axes[1].set_title("F1 Score Curves (Improved)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{filename_prefix}.png", dpi=150)
    plt.close(fig)

def plot_confusion_matrix(cm, filename="confusion_matrix_v2.png"):
    labels = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix (Improved)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)

def save_classification_report_csv(clf_report, path):
    rows = []
    for label_key in sorted(LABEL_NAMES):
        label_name = LABEL_NAMES[label_key]
        r = clf_report[label_name]
        rows.append({
            "class": label_name,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1-score": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })
    rows.append({
        "class": "macro avg",
        "precision": round(clf_report["macro avg"]["precision"], 4),
        "recall": round(clf_report["macro avg"]["recall"], 4),
        "f1-score": round(clf_report["macro avg"]["f1-score"], 4),
        "support": int(clf_report["macro avg"]["support"]),
    })
    rows.append({
        "class": "weighted avg",
        "precision": round(clf_report["weighted avg"]["precision"], 4),
        "recall": round(clf_report["weighted avg"]["recall"], 4),
        "f1-score": round(clf_report["weighted avg"]["f1-score"], 4),
        "support": int(clf_report["weighted avg"]["support"]),
    })
    pd.DataFrame(rows).to_csv(path, index=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  CrisisMind — Improved Multimodal Fusion Training")
    print("=" * 70)

    device = torch.device(DEVICE)
    print(f"\nDevice: {device}")

    # ── 1. Load data ──
    print("\n[1] Loading multimodal dataset...")
    df = load_multimodal_data()
    print(f"  Total multimodal samples: {len(df)}")

    y = df["label"].values
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y[train_idx]
    )
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    dist = train_df["label"].value_counts().sort_index()
    dist.index = dist.index.map(LABEL_NAMES)
    for label, count in dist.items():
        print(f"    {label:30s}  {count}")

    # ── 2. Datasets ──
    print("\n[2] Creating datasets with class-aware augmentations...")
    train_ds = ImprovedMultimodalDataset(train_df, is_train=True)
    val_ds = ImprovedMultimodalDataset(val_df, is_train=False)
    test_ds = ImprovedMultimodalDataset(test_df, is_train=False)

    # ── 3. WeightedRandomSampler ──
    print("\n[3] Creating WeightedRandomSampler...")
    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels), replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=FUSION_BATCH_SIZE, sampler=sampler,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=FUSION_BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=FUSION_BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── 4. Models ──
    print("\n[4] Loading models...")

    img_encoder = ImageFeatureExtractor(IMG_CKPT, freeze=False).to(device)
    txt_encoder = TextFeatureExtractor(TEXT_CKPT, freeze=True).to(device)
    img_dim = img_encoder.out_dim
    txt_dim = txt_encoder.out_dim
    print(f"  Image encoder dim: {img_dim}")
    print(f"  Text encoder dim:  {txt_dim}")

    # Unfreeze only last 2 EfficientNet stages (features.7 + features.8)
    print("  Freezing all EfficientNet stages except features.7 and features.8...")
    for name, param in img_encoder.backbone.named_parameters():
        if name.startswith("features.7") or name.startswith("features.8"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_img = sum(p.numel() for p in img_encoder.parameters() if p.requires_grad)
    total_img = sum(p.numel() for p in img_encoder.parameters())
    print(f"    Image encoder: {trainable_img:,} trainable / {total_img:,} total")

    gated = GatedFusion(
        img_dim=img_dim, txt_dim=txt_dim,
        num_classes=NUM_CLASSES, dropout=FUSION_DROPOUT,
    ).to(device)
    t, tt = count_parameters(gated)
    print(f"  GatedFusion: {t:,} trainable / {tt:,} total")

    # ── 5. Loss ──
    print("\n[5] Loss: FocalLoss with class weights...")
    class_weights_tensor = compute_class_weights(train_df).to(device)
    print(f"  Class weights: {class_weights_tensor.tolist()}")
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    print(f"  FocalLoss: alpha={class_weights_tensor.tolist()}, gamma=2.0")

    # ── 6. Optimizer ──
    trainable_params = (
        list(gated.parameters())
        + [p for p in img_encoder.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=FUSION_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, verbose=True,
    )
    early_stopping = EarlyStopping(patience=FUSION_PATIENCE, verbose=True)

    # ── 7. Gate monitor ──
    gate_monitor = GateMonitor()

    # ── 8. Training loop ──
    N_EPOCHS = 15
    print(f"\n[6] Training for up to {N_EPOCHS} epochs...")
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_f1, train_gates = train_epoch(
            gated, img_encoder, txt_encoder, train_loader, criterion, optimizer, device,
        )
        val_loss, val_f1, val_gates, val_labels, val_preds = validate_epoch(
            gated, img_encoder, txt_encoder, val_loader, criterion, device,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        gate_monitor.record(epoch, val_gates)
        avg_img_gate = val_gates[:, 0].mean().item()
        avg_txt_gate = val_gates[:, 1].mean().item()

        print(
            f"  Epoch {epoch:2d}/{N_EPOCHS}  "
            f"Train Loss: {train_loss:.4f}  Train F1: {train_f1:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val F1: {val_f1:.4f}  "
            f"Gates [img={avg_img_gate:.4f}, txt={avg_txt_gate:.4f}]"
        )

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = CHECKPOINTS_DIR / "best_gated_fusion_v2.pth"
            torch.save(gated.state_dict(), ckpt_path)
            print(f"  --> New best saved (Val Macro F1: {val_f1:.4f})")

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    # ── 9. Save gate data ──
    print("\n[7] Saving gate statistics...")
    gate_monitor.save_csv(METRICS_DIR / "gate_statistics.csv")
    gate_monitor.save_plot(PLOTS_DIR / "gate_evolution.png")
    print(f"  CSV: {METRICS_DIR / 'gate_statistics.csv'}")
    print(f"  Plot: {PLOTS_DIR / 'gate_evolution.png'}")

    # ── 10. Plot training curves ──
    plot_training_history(history, filename_prefix="training_curves_v2")
    print(f"  Training curves: {PLOTS_DIR / 'training_curves_v2.png'}")

    # ── 11. Load best checkpoint and evaluate ──
    print("\n[8] Loading best checkpoint and evaluating on test set...")
    gated.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_gated_fusion_v2.pth", map_location=device))
    metrics, clf_report, cm, y_true, y_pred = evaluate_model(
        gated, img_encoder, txt_encoder, test_loader, device,
    )
    print(f"  Test Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Test Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")

    # ── 12. Save evaluation artifacts ──
    plot_confusion_matrix(cm, filename="confusion_matrix_v2.png")
    save_classification_report_csv(clf_report, METRICS_DIR / "classification_report_v2.csv")
    print(f"  Confusion matrix: {PLOTS_DIR / 'confusion_matrix_v2.png'}")
    print(f"  Classification report: {METRICS_DIR / 'classification_report_v2.csv'}")

    # Print per-class metrics
    print("\n  Per-class metrics:")
    for label_key in sorted(LABEL_NAMES):
        name = LABEL_NAMES[label_key]
        r = clf_report[name]
        print(f"    {name:30s}  prec={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1-score']:.4f}  support={int(r['support'])}")

    # ── 13. Sanity tests ──
    sanity_df = print_sanity_results(gated, img_encoder, txt_encoder, test_df, device, n=10)
    sanity_df.to_csv(METRICS_DIR / "sanity_checks_v2.csv", index=False)
    print(f"\n  Sanity results: {METRICS_DIR / 'sanity_checks_v2.csv'}")

    # ── 14. Compare with previous model ──
    print("\n[9] Previous model metrics (from fusion_comparison_report.md):")
    prev_metrics = {"accuracy": 0.6352, "f1_macro": 0.5375, "f1_weighted": 0.6415}
    print(f"  Old GatedFusion: Accuracy={prev_metrics['accuracy']:.4f}, Macro F1={prev_metrics['f1_macro']:.4f}, Weighted F1={prev_metrics['f1_weighted']:.4f}")
    print(f"  New GatedFusion: Accuracy={metrics['accuracy']:.4f}, Macro F1={metrics['f1_macro']:.4f}, Weighted F1={metrics['f1_weighted']:.4f}")

    improvements = {
        "accuracy_delta": metrics["accuracy"] - prev_metrics["accuracy"],
        "f1_macro_delta": metrics["f1_macro"] - prev_metrics["f1_macro"],
        "f1_weighted_delta": metrics["f1_weighted"] - prev_metrics["f1_weighted"],
    }
    print(f"  Delta: Accuracy={improvements['accuracy_delta']:+.4f}, Macro F1={improvements['f1_macro_delta']:+.4f}, Weighted F1={improvements['f1_weighted_delta']:+.4f}")

    # Save comparison
    comparison = pd.DataFrame([{
        "model": "GatedFusion v1 (previous)",
        "accuracy": prev_metrics["accuracy"],
        "f1_macro": prev_metrics["f1_macro"],
        "f1_weighted": prev_metrics["f1_weighted"],
    }, {
        "model": "GatedFusion v2 (improved)",
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
    }])
    comparison.to_csv(METRICS_DIR / "model_comparison_v2.csv", index=False)
    print(f"  Comparison: {METRICS_DIR / 'model_comparison_v2.csv'}")

    print("\n" + "=" * 70)
    print("  Improved fusion training complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
