import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from backend.training.config import LABEL_NAMES


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

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
    clf_report = classification_report(y_true, y_pred,
                                        target_names=target_names,
                                        output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return metrics, clf_report, cm, y_true, y_pred


def save_metrics_csv(metrics, path):
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def save_classification_report(clf_report, path):
    rows = []
    for label, scores in clf_report.items():
        if isinstance(scores, dict):
            row = {"label": label}
            row.update(scores)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
