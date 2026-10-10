import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from backend.training.config import IMG_SIZE, BATCH_SIZE, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class CrisisDamageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        path = row["full_path"]
        label = int(row["label"])

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        if self.transform:
            image = self.transform(image)

        return image, label


def create_loaders(train_df, val_df, test_df, batch_size=BATCH_SIZE):
    set_seed()
    train_ds = CrisisDamageDataset(train_df, transform=train_transform)
    val_ds = CrisisDamageDataset(val_df, transform=val_transform)
    test_ds = CrisisDamageDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader
