import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from backend.training.config import TEXT_MODEL_NAME, TEXT_MAX_LEN, TEXT_BATCH_SIZE, RANDOM_SEED
import numpy as np


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


class CrisisTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_len=TEXT_MAX_LEN):
        self.texts = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        self.tokenizer = tokenizer or DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return input_ids, attention_mask, labels


def create_text_loaders(train_df, val_df, test_df, batch_size=TEXT_BATCH_SIZE):
    set_seed()
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)

    train_ds = CrisisTextDataset(
        train_df["tweet_text"].values, train_df["label"].values,
        tokenizer=tokenizer
    )
    val_ds = CrisisTextDataset(
        val_df["tweet_text"].values, val_df["label"].values,
        tokenizer=tokenizer
    )
    test_ds = CrisisTextDataset(
        test_df["tweet_text"].values, test_df["label"].values,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    return train_loader, val_loader, test_loader
