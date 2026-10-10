import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from backend.training.config import (
    TEXT_LEARNING_RATE, TEXT_PATIENCE, TEXT_EPOCHS, CHECKPOINTS_DIR
)


class EarlyStopping:
    def __init__(self, patience=TEXT_PATIENCE, verbose=True):
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


def compute_class_weights(train_df, num_classes=3):
    counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_text_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, f1


def validate_text_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, f1


def run_text_training(model, train_loader, val_loader, train_df, device):
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_df).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEXT_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=TEXT_EPOCHS
    )
    early_stopping = EarlyStopping(patience=TEXT_PATIENCE)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, TEXT_EPOCHS + 1):
        train_loss, train_f1 = train_text_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = validate_text_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {epoch:2d}/{TEXT_EPOCHS}  "
              f"Train Loss: {train_loss:.4f}  Train F1: {train_f1:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val F1: {val_f1:.4f}")

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best_text_model.pth")
            print(f"  --> New best text model saved (Val F1: {val_f1:.4f})")

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    return history
