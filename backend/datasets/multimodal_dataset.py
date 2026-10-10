import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from torchvision import transforms
from backend.training.config import IMG_SIZE, TEXT_BATCH_SIZE, TEXT_MODEL_NAME, RANDOM_SEED


IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer=None, max_len=64):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer or DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.max_len = max_len

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
        image = IMG_TRANSFORM(image)

        encoded = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
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
