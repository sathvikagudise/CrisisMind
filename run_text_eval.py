import torch
import numpy as np
import pandas as pd
import glob
import os
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

from backend.training.config import (
    ANNOTATIONS_DIR, LABEL_MAP, LABEL_NAMES,
    RANDOM_SEED, TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT,
    CHECKPOINTS_DIR, METRICS_DIR, REPORTS_DIR, PLOTS_DIR,
)
from backend.datasets.text_preprocessing import clean_tweet
from backend.datasets.text_dataset import create_text_loaders
from backend.models.text_model import build_text_model, count_parameters
from backend.training.text_evaluate import evaluate_text, save_metrics_csv, save_classification_report
from backend.training.visualize import plot_confusion_matrix


def load_and_prepare():
    tsv_files = sorted(glob.glob(str(ANNOTATIONS_DIR / "*.tsv")))
    tsv_files = [f for f in tsv_files if not os.path.basename(f).startswith("._")]
    rows = []
    for fp in tsv_files:
        df = pd.read_csv(fp, sep="\t", keep_default_na=True,
                         na_values=["", "nan", "NaN", None])
        sub = df[["tweet_text", "image_damage"]].copy()
        sub.dropna(subset=["image_damage"], inplace=True)
        sub = sub[sub["image_damage"] != "dont_know_or_cant_judge"]
        sub = sub[sub["image_damage"].isin(LABEL_MAP)]
        sub["label"] = sub["image_damage"].map(LABEL_MAP)
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)
    combined["tweet_text"] = combined["tweet_text"].fillna("")
    combined["tweet_text"] = combined["tweet_text"].apply(clean_tweet)
    combined.drop_duplicates(subset=["tweet_text"], keep="first", inplace=True)
    combined = combined[combined["tweet_text"].str.len() > 0].copy()
    return combined


def main():
    print("=" * 60)
    print("  Text Baseline — Evaluation Only")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\n[1] Loading data...")
    df = load_and_prepare()
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

    print("\n[2] Creating test loader...")
    _, _, test_loader = create_text_loaders(train_df, val_df, test_df)

    print("\n[3] Loading best model checkpoint...")
    model = build_text_model(num_classes=len(LABEL_MAP), freeze_bert=False)
    best_path = CHECKPOINTS_DIR / "best_text_model.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))
    model = model.to(device)
    print(f"  Loaded: {best_path}")

    print("\n[4] Evaluating on test set...")
    metrics, clf_report, cm, y_true, y_pred = evaluate_text(model, test_loader, device)

    print(f"\n  Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score (weighted):  {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):    {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):       {metrics['recall_macro']:.4f}")
    print(f"  F1 Score (macro):     {metrics['f1_macro']:.4f}")

    print("\n[5] Saving outputs...")
    save_metrics_csv(metrics, METRICS_DIR / "text_metrics.csv")
    save_classification_report(clf_report, METRICS_DIR / "text_classification_report.csv")
    plot_confusion_matrix(cm, filename="text_confusion_matrix.png")

    print(f"\n  Metrics CSV:       {METRICS_DIR / 'text_metrics.csv'}")
    print(f"  Class report CSV:  {METRICS_DIR / 'text_classification_report.csv'}")
    print(f"  Confusion matrix:  {PLOTS_DIR / 'text_confusion_matrix.png'}")

    print("\n[6] Generating evaluation summary report...")
    dist = df["label"].value_counts().sort_index()
    dist.index = dist.index.map(LABEL_NAMES)

    with open(REPORTS_DIR / "text_evaluation_summary.md", "w") as f:
        f.write("# Text Baseline Model — Evaluation Summary\n\n")
        f.write("**Model:** DistilBERT (distilbert-base-uncased)\n\n")
        f.write("## Dataset\n\n")
        f.write(f"- Total unique text samples: {len(df)}\n")
        f.write(f"- Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}\n\n")
        f.write("### Class Distribution\n\n")
        f.write("| Label | Count |\n")
        f.write("|---|---|\n")
        for label, count in dist.items():
            f.write(f"| {label} | {count} |\n")
        f.write("\n")
        f.write("## Test Set Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision (weighted) | {metrics['precision_weighted']:.4f} |\n")
        f.write(f"| Recall (weighted) | {metrics['recall_weighted']:.4f} |\n")
        f.write(f"| F1 Score (weighted) | {metrics['f1_weighted']:.4f} |\n")
        f.write(f"| Precision (macro) | {metrics['precision_macro']:.4f} |\n")
        f.write(f"| Recall (macro) | {metrics['recall_macro']:.4f} |\n")
        f.write(f"| F1 Score (macro) | {metrics['f1_macro']:.4f} |\n\n")

        f.write("## Per-Class Performance\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        for label_key in sorted(LABEL_NAMES):
            label_name = LABEL_NAMES[label_key]
            if label_name in clf_report:
                r = clf_report[label_name]
                f.write(f"| {label_name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |\n")
        f.write("\n")
        f.write("## Visualizations\n\n")
        f.write("![Confusion Matrix](../outputs/plots/text_confusion_matrix.png)\n\n")

    print(f"  Evaluation report: {REPORTS_DIR / 'text_evaluation_summary.md'}")

    print("\n" + "=" * 60)
    print("  Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
