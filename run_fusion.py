import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from backend.training.config import (
    LABEL_MAP, RANDOM_SEED, TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT,
    CHECKPOINTS_DIR, FUSION_BATCH_SIZE, FUSION_EPOCHS, FUSION_LR,
    FUSION_PATIENCE, FUSION_DROPOUT, METRICS_DIR, PLOTS_DIR, REPORTS_DIR,
    IMG_CKPT, TEXT_CKPT,
)
from backend.datasets.multimodal_data import load_multimodal_data
from backend.datasets.multimodal_dataset import MultimodalDataset, collate_fn
from backend.models.fusion_models import (
    ImageFeatureExtractor, TextFeatureExtractor,
    ConcatFusion, GatedFusion, count_parameters,
)
from backend.training.fusion_trainer import train_fusion
from backend.training.fusion_evaluate import evaluate_fusion
from backend.training.visualize import (
    plot_training_history, plot_confusion_matrix, plot_class_distribution, plot_split_counts
)
from backend.training.fusion_comparison import build_comparison_table, save_comparison_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def extract_and_cache(df, img_encoder, txt_encoder, device, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "features.pt"
    if cache_path.exists():
        print("  Loading cached features...")
        data = torch.load(cache_path)
        return data

    ds = MultimodalDataset(df)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)

    all_img_feat, all_txt_feat, all_labels = [], [], []

    print(f"  Extracting features from {len(df)} samples...")
    img_encoder.eval()
    txt_encoder.eval()
    with torch.no_grad():
        for i, (images, input_ids, attention_mask, labels) in enumerate(loader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            img_feat = img_encoder(images)
            txt_feat = txt_encoder(input_ids, attention_mask)

            all_img_feat.append(img_feat.cpu())
            all_txt_feat.append(txt_feat.cpu())
            all_labels.append(labels.cpu())

            if (i + 1) % 20 == 0:
                print(f"    Processed {i + 1}/{len(loader)} batches")

    data = {
        "img_feat": torch.cat(all_img_feat, dim=0),
        "txt_feat": torch.cat(all_txt_feat, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }
    torch.save(data, cache_path)
    print(f"  Cached features to {cache_path}")
    return data


from pathlib import Path


class CachedFusionDataset(Dataset):
    def __init__(self, data):
        self.img_feat = data["img_feat"]
        self.txt_feat = data["txt_feat"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "img_feat": self.img_feat[idx],
            "txt_feat": self.txt_feat[idx],
            "label": self.labels[idx],
        }


def collate_fusion(batch):
    img_feat = torch.stack([b["img_feat"] for b in batch])
    txt_feat = torch.stack([b["txt_feat"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return img_feat, txt_feat, labels


def get_cached_loader(data, batch_size=FUSION_BATCH_SIZE, shuffle=False):
    ds = CachedFusionDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fusion)


def train_fusion_cached(model_name, model, train_loader, val_loader, train_df, device):
    import torch.nn as nn
    from backend.training.fusion_trainer import EarlyStopping
    from sklearn.metrics import f1_score

    counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum() * 3
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=FUSION_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=FUSION_PATIENCE)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, FUSION_EPOCHS + 1):
        model.train()
        train_loss, train_preds, train_lbls = 0.0, [], []
        for img_feat, txt_feat, labels in train_loader:
            img_feat, txt_feat, labels = img_feat.to(device), txt_feat.to(device), labels.to(device)
            if hasattr(model, 'gate_net'):
                logits, _ = model(img_feat, txt_feat)
            else:
                logits = model(img_feat, txt_feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img_feat.size(0)
            _, preds = torch.max(logits, 1)
            train_preds.extend(preds.cpu().numpy())
            train_lbls.extend(labels.cpu().numpy())
        train_loss /= len(train_loader.dataset)
        train_f1 = f1_score(train_lbls, train_preds, average="macro")

        model.eval()
        val_loss, val_preds, val_lbls = 0.0, [], []
        with torch.no_grad():
            for img_feat, txt_feat, labels in val_loader:
                img_feat, txt_feat, labels = img_feat.to(device), txt_feat.to(device), labels.to(device)
                if hasattr(model, 'gate_net'):
                    logits, _ = model(img_feat, txt_feat)
                else:
                    logits = model(img_feat, txt_feat)
                loss = criterion(logits, labels)
                val_loss += loss.item() * img_feat.size(0)
                _, preds = torch.max(logits, 1)
                val_preds.extend(preds.cpu().numpy())
                val_lbls.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_lbls, val_preds, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {epoch:2d}/{FUSION_EPOCHS}  "
              f"Train Loss: {train_loss:.4f}  Train F1: {train_f1:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val F1: {val_f1:.4f}")

        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"best_{model_name}.pth")
            print(f"  --> New best saved (Val Macro F1: {val_f1:.4f})")

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    return history


def evaluate_fusion_cached(model, loader, device):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img_feat, txt_feat, labels in loader:
            img_feat, txt_feat, labels = img_feat.to(device), txt_feat.to(device), labels.to(device)
            if hasattr(model, 'gate_net'):
                logits, _ = model(img_feat, txt_feat)
            else:
                logits = model(img_feat, txt_feat)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
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


from backend.training.config import LABEL_NAMES
import numpy as np
import pandas as pd


def main():
    print("=" * 60)
    print("  CrisisMind — Multimodal Fusion (Phase 3)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\n[1] Loading multimodal dataset...")
    df = load_multimodal_data()
    print(f"  Total multimodal samples: {len(df)}")

    dist = df["label"].value_counts().sort_index()
    dist.index = dist.index.map(LABEL_NAMES)
    for label, count in dist.items():
        print(f"  {label:30s}  {count}")
    plot_class_distribution(dist, "Multimodal Class Distribution", "fusion_class_distribution.png")

    print("\n[2] Creating stratified splits...")
    y = df["label"].values
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y[train_idx])
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    plot_split_counts(len(train_df), len(val_df), len(val_df), filename="fusion_split_distribution.png")

    print("\n[3] Loading feature extractors...")
    img_encoder = ImageFeatureExtractor(IMG_CKPT, freeze=True).to(device)
    txt_encoder = TextFeatureExtractor(TEXT_CKPT, freeze=True).to(device)
    img_dim = img_encoder.out_dim
    txt_dim = txt_encoder.out_dim
    print(f"  Image encoder dim: {img_dim}")
    print(f"  Text encoder dim:  {txt_dim}")

    print("\n[4] Extracting features (cached)...")
    cache_dir = Path("backend/outputs/cached_features")
    train_data = extract_and_cache(train_df, img_encoder, txt_encoder, device, cache_dir / "train")
    val_data = extract_and_cache(val_df, img_encoder, txt_encoder, device, cache_dir / "val")
    test_data = extract_and_cache(test_df, img_encoder, txt_encoder, device, cache_dir / "test")

    train_loader = get_cached_loader(train_data, shuffle=True)
    val_loader = get_cached_loader(val_data)
    test_loader = get_cached_loader(test_data)

    results = {}
    clf_reports = {}

    # ── Fusion A: Concat ──
    print("\n" + "=" * 50)
    print("  Model A: Concat Fusion")
    print("=" * 50)
    concat = ConcatFusion(img_dim, txt_dim, num_classes=len(LABEL_MAP), dropout=FUSION_DROPOUT).to(device)
    t, tt = count_parameters(concat)
    print(f"  Trainable params: {t:,} / Total: {tt:,}")
    hist_c = train_fusion_cached("concat_fusion", concat, train_loader, val_loader, train_df, device)
    concat.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_concat_fusion.pth", map_location=device))
    m, cr, cm_mat, _, _ = evaluate_fusion_cached(concat, test_loader, device)
    results["ConcatFusion"] = m
    clf_reports["ConcatFusion"] = cr
    plot_confusion_matrix(cm_mat, filename="concat_fusion_cm.png")
    plot_training_history(hist_c, filename_prefix="concat_fusion")
    pd.DataFrame([m]).to_csv(METRICS_DIR / "concat_fusion_metrics.csv", index=False)
    print(f"  Test Macro F1: {m['f1_macro']:.4f}")

    # ── Fusion B: Gated ──
    print("\n" + "=" * 50)
    print("  Model B: Gated Fusion")
    print("=" * 50)
    gated = GatedFusion(img_dim, txt_dim, num_classes=len(LABEL_MAP), dropout=FUSION_DROPOUT).to(device)
    t, tt = count_parameters(gated)
    print(f"  Trainable params: {t:,} / Total: {tt:,}")
    hist_g = train_fusion_cached("gated_fusion", gated, train_loader, val_loader, train_df, device)
    gated.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_gated_fusion.pth", map_location=device))
    m_g, cr_g, cm_g, _, _ = evaluate_fusion_cached(gated, test_loader, device)
    results["GatedFusion"] = m_g
    clf_reports["GatedFusion"] = cr_g
    plot_confusion_matrix(cm_g, filename="gated_fusion_cm.png")
    plot_training_history(hist_g, filename_prefix="gated_fusion")
    pd.DataFrame([m_g]).to_csv(METRICS_DIR / "gated_fusion_metrics.csv", index=False)
    print(f"  Test Macro F1: {m_g['f1_macro']:.4f}")

    # ── Load baselines ──
    for name, pth in [("ImageBaseline", METRICS_DIR / "test_metrics.csv"),
                      ("TextBaseline", METRICS_DIR / "text_metrics.csv")]:
        if pth.exists():
            results[name] = {k: float(v) for k, v in pd.read_csv(pth).iloc[0].items()}

    print("\n[5] Comparison:")
    comp_df, _ = build_comparison_table(results)
    print(comp_df.to_string(index=False))
    save_comparison_report(results, clf_reports, REPORTS_DIR / "fusion_comparison_report.md")
    print(f"\n  Report: {REPORTS_DIR / 'fusion_comparison_report.md'}")

    print("\n" + "=" * 60)
    print("  Multimodal fusion complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
