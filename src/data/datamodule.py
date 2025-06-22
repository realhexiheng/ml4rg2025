from pathlib import Path
from abc import ABC, abstractmethod

import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


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


class GeneDataset(Dataset):
    """Simple Dataset wrapper for gene data."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()


class CrossValidationDataModule(L.LightningDataModule, ABC):
    """
    Abstract Lightning DataModule for DNA embeddings with cross-validation.
    """

    def __init__(
        self,
        reader: HDFReader,
        summary: str | Path,
        test_fold: int,
        validation: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Initialize CrossValidationDataModule.

        Args:
            reader: HDFReader instance for loading embeddings/expressions
            summary: Path to summary CSV file
            test_fold: Which fold (0-4) to use as test set
            validation: Whether to split training data into train/val
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.save_hyperparameters()

        self.reader = reader
        self.test_fold = test_fold
        self.validation = validation
        self.seed = seed

        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "prefetch_factor": 10,
            "persistent_workers": True,
            "num_workers": num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": True,
        }

        if not 0 <= test_fold <= 4:
            raise ValueError(f"Index of fold must be 0-4, got {test_fold}")

        # Load and prepare data
        self.summary = pd.read_csv(summary)
        self._create_folds()
        self._create_splits()

    @abstractmethod
    def _create_folds(self):
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

    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Custom collate function to load embeddings (X) and RNA targets (y) on-demand.

        Args:
            batch: List of dictionaries from dataset

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


class ChromosomeStratifiedDataModule(CrossValidationDataModule):
    """
    CrossValidationDataModule with chromosome-wise stratification.
    """

    def _create_folds(self):
        """Create fold assignments using chromosome stratification."""
        from sklearn.model_selection import StratifiedKFold

        self.summary["fold"] = -1

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.seed,
        )

        for fold, (_, test_idx) in enumerate(
            skf.split(
                X=self.summary["gene"],
                y=self.summary["chromosome"],
            )
        ):
            self.summary.loc[test_idx, "fold"] = fold

        print("\nChromosome distribution by fold:")
        print(self.summary.groupby(["fold", "chromosome"]).size().unstack(fill_value=0))


class ParalogousGeneDataModule(CrossValidationDataModule):
    """
    CrossValidationDataModule with chromosome and paralog group stratification.
    """

    def _create_folds(self):
        """Create fold assignments using chromosome and paralog group stratification."""
        from sklearn.model_selection import StratifiedGroupKFold

        if "paralog_group" not in self.summary.columns:
            raise ValueError(
                "Summary CSV must contain 'paralog_group' column with paralog information. "
                "Please run preprocessing with include_paralogs=True."
            )

        self.summary["fold"] = -1

        sgkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.seed,
        )

        for fold, (_, test_idx) in enumerate(
            sgkf.split(
                self.summary["gene"],
                self.summary["chromosome"],
                groups=self.summary["paralog_group"],
            )
        ):
            self.summary.loc[test_idx, "fold"] = fold

        print("\nStratifiedGroupKFold chromosome distribution by fold:")
        print(self.summary.groupby(["fold", "chromosome"]).size().unstack(fill_value=0))
