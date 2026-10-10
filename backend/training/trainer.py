import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from backend.training.config import CHECKPOINTS_DIR, EPOCHS, LEARNING_RATE, PATIENCE


class EarlyStopping:
    def __init__(self, patience=PATIENCE, verbose=True):
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
        else:
            self.best_score = score
            self.counter = 0
            return False


def compute_class_weights(train_df, num_classes=3):
    counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, f1


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, f1


def run_training(model, train_loader, val_loader, train_df, device):
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_df).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True
    )
    early_stopping = EarlyStopping(patience=PATIENCE)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"Train Loss: {train_loss:.4f}  Train F1: {train_f1:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best_model.pth")
            print(f"  --> New best model saved (Val F1: {val_f1:.4f})")

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    return history
