"""
data.py — Dataset loading and preprocessing
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Vocabulary mapping
VOCAB = {"PAD": 0, "A": 1, "B": 2, "C": 3, "D": 4}
VOCAB_SIZE = 5
MAX_LEN = 20

# Column names in the CSV
TOKEN_COLS = [f"token_{i:02d}" for i in range(1, 21)]
MASK_COLS = [f"mask_{i:02d}" for i in range(1, 21)]


class TransformerDataset(Dataset):
    """Loads token sequences, attention masks, and labels from a DataFrame."""

    def __init__(self, df):
        self.tokens = df[TOKEN_COLS].values.astype(np.int64)
        self.masks = df[MASK_COLS].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokens[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tokens, mask, label


def get_dataloaders(data_dir, batch_size=32):
    """Load train/val/test CSVs and return PyTorch DataLoaders."""
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/validation.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_ds = TransformerDataset(train_df)
    val_ds = TransformerDataset(val_df)
    test_ds = TransformerDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader