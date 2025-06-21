from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader


class HDFReader:
    """Reader class for HDF5 files containing chromosome-wise gene embeddings and expressions."""

    def __init__(self, hdf: str | Path):
        """
        Initialize HDFReader with path to HDF5 file.

        Args:
            hdf (str | Path): Path to the HDF5 file
        """
        self.data = h5py.File(hdf, "r")

    def __getitem__(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings and expressions for a given chromosome and gene indices.

        Args:
            key (str): CDS indentifier.

        Returns:
            (np.ndarray, np.ndarray): Tuple containing gene embeddings and normalized expressions.
        """
        embeddings = self.data[key]["embedding"][:]
        expressions = self.data[key]["expression"][:]

        return embeddings, expressions


class CrossValidationDataModule(L.LightningDataModule):
    """
    Lightning DataModule for DNA embeddings with chromosome-wise cross-validation.
    Uses HDF5 file for embeddings/expressions and summary CSV for fold information.
    """

    def __init__(
        self,
        hdf: str | Path,
        dataset: str | Path,
        test_fold: int,
        validation: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Initialize DNADataModule.

        Args:
            hdf: Path to HDF5 file containing gene embeddings and expressions
            dataset: Path to saved HuggingFace dataset directory
            test_fold: Which fold (0-4) to use as test set
            validation: Whether to split training data into train/val
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.save_hyperparameters()

        self.test_fold = test_fold
        self.validation = validation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        if not 0 <= test_fold <= 4:
            raise ValueError(f"Index of fold must be 0-4, got {test_fold}")

        self.dataset = load_from_disk(dataset)
        self.reader = HDFReader(hdf)

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str | None = None):
        """
        Setup data for training/validation/testing.
        """

        if "fold" not in self.dataset.column_names:
            raise ValueError("Dataset must contain 'fold' column")

        test_data = self.dataset.filter(lambda x: x["fold"] == self.test_fold)
        train_data = self.dataset.filter(lambda x: x["fold"] != self.test_fold)
        val_data = None

        if self.validation:
            splits = train_data.train_test_split(
                test_size=0.2,
                stratify_by_column="fold",
                seed=self.seed,
            )

            train_data = splits["train"]
            val_data = splits["test"]

        if stage == "fit":
            self.train_data = train_data
            self.val_data = val_data if self.validation else None

        if stage == "test":
            self.test_data = test_data

    def train_dataloader(self):
        """Create training data loader."""
        if self.train_data is None:
            raise RuntimeError("train_data is None. Call setup() first.")

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if not self.validation or self.val_data is None:
            return None

        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test data loader."""
        if self.test_data is None:
            raise RuntimeError("test_data is None. Call setup() first.")

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Custom collate function to load embeddings (X) and RNA targets (y) on-demand.

        Args:
            batch: List of dictionaries from HuggingFace dataset

        Returns:
            Dictionary with 'embeddings' and 'expressions' tensors
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

        return {
            "embeddings": embeddings,  # Shape: (batch_size, 500, 768)
            "expressions": expressions,  # Shape: (batch_size, 18)
        }
