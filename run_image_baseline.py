import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split

from backend.training.config import (
    LABEL_NAMES, LABEL_MAP, RANDOM_SEED,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, BATCH_SIZE,
    CHECKPOINTS_DIR, METRICS_DIR, REPORTS_DIR, PLOTS_DIR,
)
from backend.datasets.prepare_data import (
    load_tsvs, prepare_damage_data, verify_images, class_distribution
)
from backend.datasets.data_loader import create_loaders
from backend.models.efficientnet import build_model, count_parameters
from backend.training.trainer import run_training
from backend.training.evaluate import evaluate, save_metrics_csv, save_classification_report
from backend.training.visualize import (
    plot_class_distribution, plot_split_counts,
    plot_training_history, plot_confusion_matrix,
)


def main():
    print("=" * 60)
    print("  CrisisMind — Image Baseline Model (Phase 1)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\n[1] Loading and preparing dataset...")
    dataframes = load_tsvs()
    df = prepare_damage_data(dataframes)
    df = verify_images(df)
    print(f"  Total damage samples: {len(df)}")

    print("\n[2] Class distribution:")
    dist = class_distribution(df)
    for label, count in dist.items():
        print(f"  {label:30s}  {count}")
    plot_class_distribution(dist, "Damage Severity Class Distribution",
                            "class_distribution.png")

    print("\n[3] Creating stratified splits...")
    y = df["label"].values
    indices = np.arange(len(df))

    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
        stratify=y
    )
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_ratio, random_state=RANDOM_SEED,
        stratify=y_train
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print(f"  Train:       {len(train_df)} samples")
    print(f"  Validation:  {len(val_df)} samples")
    print(f"  Test:        {len(test_df)} samples")

    plot_split_counts(len(train_df), len(val_df), len(test_df))

    print("\n[4] Creating data loaders...")
    train_loader, val_loader, test_loader = create_loaders(
        train_df, val_df, test_df, batch_size=BATCH_SIZE
    )

    print("\n[5] Building EfficientNet-B0 model...")
    model = build_model(num_classes=len(LABEL_MAP), freeze_backbone=True)
    trainable, total = count_parameters(model)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Total params:     {total:,}")
    model = model.to(device)

    print("\n[6] Training...")
    history = run_training(model, train_loader, val_loader, train_df, device)

    print("\n[7] Loading best model for evaluation...")
    best_path = CHECKPOINTS_DIR / "best_model.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    print("\n[8] Evaluation on test set...")
    metrics, clf_report, cm, y_true, y_pred = evaluate(model, test_loader, device)

    print(f"\n  Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score (weighted):  {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):    {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):       {metrics['recall_macro']:.4f}")
    print(f"  F1 Score (macro):     {metrics['f1_macro']:.4f}")

    print("\n[9] Saving outputs...")
    save_metrics_csv(metrics, METRICS_DIR / "test_metrics.csv")
    save_classification_report(clf_report, METRICS_DIR / "classification_report.csv")
    plot_confusion_matrix(cm)
    plot_training_history(history)

    print(f"\n  Best model:        {CHECKPOINTS_DIR / 'best_model.pth'}")
    print(f"  Metrics CSV:       {METRICS_DIR / 'test_metrics.csv'}")
    print(f"  Class report CSV:  {METRICS_DIR / 'classification_report.csv'}")
    print(f"  Confusion matrix:  {PLOTS_DIR / 'confusion_matrix.png'}")
    print(f"  Training curves:   {PLOTS_DIR / 'training_curves.png'}")

    print("\n[10] Generating evaluation summary report...")
    with open(REPORTS_DIR / "evaluation_summary.md", "w") as f:
        f.write("# Image Baseline Model — Evaluation Summary\n\n")
        f.write(f"**Model:** EfficientNet-B0 (pretrained on ImageNet)\n\n")
        f.write(f"## Dataset\n\n")
        f.write(f"- Total damage samples: {len(df)}\n")
        f.write(f"- Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}\n\n")
        f.write("### Class Distribution\n\n")
        f.write("| Label | Count |\n")
        f.write("|---|---|\n")
        for label, count in dist.items():
            f.write(f"| {label} | {count} |\n")
        f.write("\n")
        f.write("## Test Set Performance\n\n")
        f.write(f"| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision (weighted) | {metrics['precision_weighted']:.4f} |\n")
        f.write(f"| Recall (weighted) | {metrics['recall_weighted']:.4f} |\n")
        f.write(f"| F1 Score (weighted) | {metrics['f1_weighted']:.4f} |\n")
        f.write(f"| Precision (macro) | {metrics['precision_macro']:.4f} |\n")
        f.write(f"| Recall (macro) | {metrics['recall_macro']:.4f} |\n")
        f.write(f"| F1 Score (macro) | {metrics['f1_macro']:.4f} |\n\n")

        f.write("## Visualization\n\n")
        f.write("![Confusion Matrix](../outputs/plots/confusion_matrix.png)\n\n")
        f.write("![Training Curves](../outputs/plots/training_curves.png)\n\n")
        f.write("![Class Distribution](../outputs/plots/class_distribution.png)\n\n")
        f.write("![Split Distribution](../outputs/plots/split_distribution.png)\n\n")

    print(f"  Evaluation report: {REPORTS_DIR / 'evaluation_summary.md'}")

    print("\n" + "=" * 60)
    print("  Image baseline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
