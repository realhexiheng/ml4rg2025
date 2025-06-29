from abc import ABC, abstractmethod

import lightning as L
import pandas as pd
import torch
from src.utils.io import HDFReader
from torch.utils.data import DataLoader, Dataset


class GeneDataset(Dataset):
    """Simple Dataset wrapper for gene data."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()

    @property
    def genes(self):
        """Return list of gene names for compatibility with model logging."""
        return self.data["gene"].tolist()


class CrossValidationDataModule(L.LightningDataModule, ABC):
    """
    Abstract Lightning DataModule for DNA embeddings with cross-validation.
    """

    def __init__(
        self,
        reader: HDFReader,
        summary: pd.DataFrame,
        test_fold: int,
        validation: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        prefetch_factor: int = 10,
        n_folds: int = 5,
    ):
        """
        Initialize CrossValidationDataModule.

        Args:
            reader: HDFReader instance for loading embeddings/expressions
            summary: Summary dataframe
            test_fold: Which fold (0-4) to use as test set
            validation: Whether to split training data into train/val
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
            prefetch_factor: Prefetch factor for data loading
        """
        super().__init__()
        self.save_hyperparameters(ignore=["reader", "summary"])

        self.reader = reader
        self.test_fold = test_fold
        self.validation = validation
        self.seed = seed

        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "num_workers": num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": torch.cuda.is_available(),
        }

        if not 0 <= test_fold <= 4:
            raise ValueError(f"Index of fold must be 0-4, got {test_fold}")

        # Load and prepare data
        self.summary = summary
        self._create_folds(n_folds)
        self._create_splits()

    @abstractmethod
    def _create_folds(self, n_folds: int):
        """Create fold assignments. Must add 'fold' column to self.summary."""
        pass

    def _create_splits(self):
        """Create train/val/test splits based on fold assignments."""
        test_data = self.summary[self.summary["fold"] == self.test_fold]
        train_data = self.summary[self.summary["fold"] != self.test_fold]
        val_data = None

        if self.validation:
            from sklearn.model_selection import train_test_split

            train_data, val_data = train_test_split(
                train_data,
                test_size=0.2,
                stratify=train_data["fold"],
                random_state=self.seed,
            )

        self.train_data = GeneDataset(train_data.reset_index(drop=True))
        self.val_data = (
            GeneDataset(val_data.reset_index(drop=True)) if val_data is not None else None
        )
        self.test_data = GeneDataset(test_data.reset_index(drop=True))

    def train_dataloader(self):
        """Create training data loader."""
        return DataLoader(
            self.train_data,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if not self.validation or self.val_data is None:
            return None

        return DataLoader(
            self.val_data,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        """Create test data loader."""
        return DataLoader(
            self.test_data,
            **self.dataloader_kwargs,
        )

    def collate_fn(self, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collate function to load embeddings (X) and RNA targets (y) on-demand.

        Args:
            batch: List of dictionaries from dataset
        Returns:
            tuple: (embeddings, expressions) where:
                - embeddings: Tensor of shape (batch_size, embedding_dim)
                - expressions: Tensor of shape (batch_size, expression_dim)

        """
        embeddings = []
        expressions = []

        for item in batch:
            gene = item["gene"]
            try:
                emb, exp = self.reader[gene]
                embeddings.append(emb)
                expressions.append(exp)
            except Exception as e:
                print(f"Warning: Failed to load data for gene {gene}: {e}")
                continue

        embeddings = torch.stack([torch.from_numpy(emb).float() for emb in embeddings])
        expressions = torch.stack([torch.from_numpy(exp).float() for exp in expressions])

        return embeddings, expressions
