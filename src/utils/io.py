from pathlib import Path

import h5py
import numpy as np


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
