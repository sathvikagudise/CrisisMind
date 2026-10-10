import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from backend.training.config import LABEL_NAMES


def evaluate_fusion(model, img_encoder, txt_encoder, loader, device):
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

            if hasattr(model, 'gate_net'):
                logits, _ = model(img_feat, txt_feat)
            else:
                logits = model(img_feat, txt_feat)

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
