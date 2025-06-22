from abc import abstractmethod
from pathlib import Path


import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value


class CustomDataset:
    """Base class for custom datasets."""

    @abstractmethod
    def create_dataset(self, outdir: str | Path) -> None:
        """
        Create the dataset by extracting gene embeddings and normalizing gene expression.
        Args:
            outdir (str | Path): Output directory to save the processed npz files.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")


class ChromosomeStratifiedDatasetSplit(CustomDataset):
    """Chromosomewise split dataset with stratification."""

    def __init__(
        self,
        summary: str | Path,
        seed: int = 42,
    ):
        """
        Initialize the ChromosomeStratifiedDatasetSplit with paths to summary and HDF files.

        Args:
            summary (str | Path): Path to the summary CSV file.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.summary = pd.read_csv(summary, usecols=["gene", "chromosome", "strand"])
        self.seed = seed

    def create_dataset(self, outdir: str | Path) -> None:
        """
        Create the dataset by extracting gene embeddings and normalizing gene expression.
        Args:
            outdir (str | Path): Output directory to save the processed HuggingFace dataset.
        """
        from sklearn.model_selection import StratifiedKFold

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

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
        self.summary["gene_index"] = self.summary.groupby("chromosome").cumcount()

        print("\nChromosome distribution by fold:")
        print(self.summary.groupby(["fold", "chromosome"]).size().unstack(fill_value=0))

        features = Features(
            {
                "gene": Value("string"),
                "chromosome": ClassLabel(
                    names=sorted(self.summary["chromosome"].unique())
                ),
                "strand": ClassLabel(names=["+", "-"]),
                "fold": ClassLabel(names=list(range(5))),
                "gene_index": Value("int32"),
            }
        )

        dataset = Dataset.from_pandas(self.summary, features=features)
        dataset.save_to_disk(outdir / "chromosome_stratified_dataset")

        return dataset


class ParalogousGeneDataset(ChromosomeStratifiedDatasetSplit):
    def __init__(
        self,
        summary: str | Path,
        seed: int = 42,
    ):
        super().__init__(summary=summary, seed=seed)

    def create_dataset(self, outdir: str | Path) -> Dataset:
        """
        Create dataset using stratified CV splitting by chromosome and grouped paralogs.
        Uses pre-computed paralog groups from the summary CSV.
        """
        from sklearn.model_selection import StratifiedGroupKFold

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

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

        self.summary["gene_index"] = self.summary.groupby("chromosome").cumcount()

        print("\nStratifiedGroupKFold chromosome distribution by fold:")
        print(self.summary.groupby(["fold", "chromosome"]).size().unstack(fill_value=0))

        features = Features(
            {
                "gene": Value("string"),
                "chromosome": ClassLabel(
                    names=sorted(self.summary["chromosome"].unique())
                ),
                "strand": ClassLabel(names=["+", "-"]),
                "fold": ClassLabel(names=list(range(5))),
                "groups": Value("string"),
                "gene_index": Value("int32"),
            }
        )

        dataset = Dataset.from_pandas(self.summary, features=features)
        dataset.save_to_disk(outdir / "paralogous_gene_dataset")

        return dataset
