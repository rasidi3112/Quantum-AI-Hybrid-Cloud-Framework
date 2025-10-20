"""Data loading and preprocessing utilities for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd # type: ignore
import torch # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from torch.utils.data import DataLoader, Dataset, TensorDataset # type: ignore


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset loading."""
    dataset_path: Path
    batch_size: int = 32
    val_split: float = 0.2
    shuffle: bool = True
    seed: int = 42


def load_tabular_dataset(config: DataConfig) -> Tuple[DataLoader, DataLoader, int]:
    """Load a tabular dataset from CSV and return dataloaders.

    Args:
        config: DataConfig specifying dataset parameters.

    Returns:
        Tuple(train_loader, val_loader, n_classes).
    """
    df = pd.read_csv(config.dataset_path)
    feature_cols = [col for col in df.columns if col != "label"]
    label_col = "label"

    features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    labels = torch.tensor(df[label_col].values, dtype=torch.long)

    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=config.val_split,
        random_state=config.seed,
        shuffle=config.shuffle,
        stratify=labels.numpy(),
    )

    train_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] = TensorDataset(
        features[train_idx], labels[train_idx]
    )
    val_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] = TensorDataset(
        features[val_idx], labels[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    n_classes = labels.unique().numel()

    return train_loader, val_loader, n_classes