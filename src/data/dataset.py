from abc import abstractmethod
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value

from src.utils.genome import (
    get_cds_coord,
    get_chromosome_lengths,
    get_chromosome_valid_genes,
    parse_gff,
)

from src.utils.expression import (
    get_gene_embeddings,
    get_normalized_gene_expression,
    load_sample_expression,
)


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


class ChromosomeDatasetSplit(CustomDataset):
    """Chromosomewise split dataset."""

    def __init__(
        self,
        gff: str | Path,
        fasta: str | Path,
        embed_dir: str | Path,
        rna_dir: str | Path,
        condition: str | Path,
    ):
        """
        Initialize the ChromosomeDataset with paths to GFF, FASTA, embeddings, RNA expression data, and condition.

        Args:
            gff (str | Path): Path to the GFF file.
            fasta (str | Path): Path to the FASTA file.
            embed_dir: str | Path: Path to the chromosome embeddings directory.
            rna_dir (str | Path): Path to the RNA expression directory (e.g. waern_2023).
            condition (str | Path): Condition for the dataset.
        """

        self.gff = Path(gff)
        self.embed_dir = Path(embed_dir)
        self.fasta = Path(fasta)
        self.rna_dir = Path(rna_dir)
        self.condition = Path(condition)

        self.folds = {
            chrom: i
            for i, chroms in enumerate(
                [
                    ["chrI", "chrII", "chrIII"],
                    ["chrIV", "chrV", "chrVI"],
                    ["chrVII", "chrIX", "chrX"],
                    ["chrXI", "chrXII", "chrXIII"],
                    ["chrXIV", "chrXV", "chrXVI"],
                ]
            )
            for chrom in chroms
        }

    def create_dataset(self, outdir: str | Path) -> None:
        """
        Create the dataset by extracting gene embeddings and normalizing gene expression.
        Args:
            outdir (str | Path, optional): Output directory to save the processed npz files.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        annotation = parse_gff(self.gff)

        chromosome_lengths = get_chromosome_lengths(self.fasta)
        cds_coords = get_cds_coord(annotation)

        valid_genes = get_chromosome_valid_genes(
            cds_coords, chromosome_lengths, window_size=500
        )

        df = pd.read_csv(self.condition, usecols=["samples", "condition"])
        df["samples"] = df["samples"].str.split(",")
        condition_with_samples = df.set_index("condition")["samples"].to_dict()
        flattened_samples = df["samples"].explode().tolist()

        summary = pd.DataFrame()

        with h5py.File(outdir / "genewise.h5", "w") as h5f:
            for chromosome, length in chromosome_lengths.items():
                print(f"Processing {chromosome}")

                valid_cds_coords = [cds_coords[gene] for gene in valid_genes[chromosome]]

                chromosome_embedding = np.load(
                    f"{self.embed_dir}/{chromosome}.npy"
                ).astype(np.float16)

                assert len(chromosome_embedding) == length, (
                    f"Length of {chromosome} embedding is {len(chromosome_embedding)} but should be {length}"
                )

                gene_embeddings = get_gene_embeddings(
                    valid_cds_coords, chromosome_embedding
                )

                del chromosome_embedding

                gene_expression = get_normalized_gene_expression(
                    valid_cds_coords,
                    condition_with_samples,
                    load_sample_expression(self.rna_dir, flattened_samples),
                )

                for i, gene in enumerate(valid_genes[chromosome]):
                    gene_group = h5f.require_group(gene)
                    gene_group.create_dataset(
                        "embedding", data=gene_embeddings[i], compression="gzip"
                    )
                    gene_group.create_dataset(
                        "expression", data=gene_expression[i], compression="gzip"
                    )

                chromosome_df = pd.DataFrame(
                    {
                        "gene": valid_genes[chromosome],
                        "coordinates": [cd["coordinates"] for cd in valid_cds_coords],
                        "chromosome": chromosome,
                        "strand": [cd["strand"] for cd in valid_cds_coords],
                        "fold": self.folds.get(chromosome, -1) + 1,
                    }
                )

                summary = pd.concat([summary, chromosome_df], ignore_index=True)

        summary.to_csv(Path(outdir) / "summary.csv", index=False)


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

    def create_clusters(
        self,
        paralogs: pd.DataFrame,
    ):
        """
        Create paralog groups using direct mapping from reciprocal relationships.

        Args:
            paralogs (pd.DataFrame): DataFrame containing paralog information
        Returns:
            dict: Gene to group_id mapping
        """

        cluster = defaultdict(set)

        paralogs = paralogs[paralogs["paralog_ensembl_id"].notna()]
        paralogs = paralogs[paralogs["paralog_ensembl_id"].isin(self.summary["gene"])]

        for _, row in paralogs.iterrows():
            gene1, gene2 = row["gene"], row["paralog_ensembl_id"]

            cluster[gene1].add(gene1)
            cluster[gene1].add(gene2)
            cluster[gene2].add(gene2)
            cluster[gene2].add(gene1)

        mapping = {}
        for representative, genes in cluster.items():
            for gene in genes:
                mapping[gene] = representative

        standalone = set(self.summary["gene"]) - set(mapping.keys())
        mapping.update({gene: gene for gene in standalone})

        return mapping

    def create_dataset(self, outdir: str | Path) -> Dataset:
        """
        Create dataset using stratified CV splitting by chromosome and grouped paralogs.
        """
        from sklearn.model_selection import StratifiedGroupKFold

        from src.utils.query import query_paralogs

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        paralogs = query_paralogs(
            self.summary["gene"].tolist(),
        )

        self.summary["fold"] = -1
        mapping = self.create_clusters(
            paralogs,
        )

        self.summary["groups"] = self.summary["gene"].map(mapping)

        sgkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.seed,
        )

        for fold, (_, test_idx) in enumerate(
            sgkf.split(
                self.summary["gene"],
                self.summary["chromosome"],
                groups=self.summary["groups"],
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
