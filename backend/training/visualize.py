import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from backend.training.config import PLOTS_DIR, LABEL_NAMES

sns.set_theme(style="whitegrid")


def plot_class_distribution(dist, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(dist))
    bars = ax.bar(dist.index, dist.values, color=colors, edgecolor="black")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Damage Severity")
    for bar, v in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(dist.values) * 0.01,
                str(v), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_split_counts(train_count, val_count, test_count, filename="split_distribution.png"):
    labels = ["Train", "Validation", "Test"]
    counts = [train_count, val_count, test_count]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("Set3", 3)
    bars = ax.bar(labels, counts, color=colors, edgecolor="black")
    ax.set_title("Train / Validation / Test Split", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(v), ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_training_history(history, filename_prefix="training"):
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
    axes[1].set_title("F1 Score Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{filename_prefix}_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, filename="confusion_matrix.png"):
    labels = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)
