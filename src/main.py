from data.dataset import (
    ChromosomeDatasetSplit,
    ChromosomeStratifiedDatasetSplit,
    ParalogousGeneDataset,
)


def create_chromosome_dataset():
    """
    Create a chromosome-wise split dataset by extracting gene embeddings and normalizing gene expression.
    """
    ChromosomeDatasetSplit(
        gff="data/gff_file.gff",
        fasta="data/fasta_file.fsa",
        embed_dir="data/all_vectors",
        rna_dir="data/waern_2013",
        condition="data/condition.csv",
    ).create_dataset("data/prepared")


def create_chromosome_stratified_dataset():
    """
    Create a chromosome-wise stratified dataset by extracting gene embeddings and normalizing gene expression.
    """
    ChromosomeStratifiedDatasetSplit(
        summary="data/summary.csv",
        seed=42,
    ).create_dataset("data/prepared")


def create_paralogous_dataset():
    ParalogousGeneDataset(
        summary="data/summary.csv",
        seed=42,
    ).create_dataset("data/prepared")


if __name__ == "__main__":
    create_paralogous_dataset()
