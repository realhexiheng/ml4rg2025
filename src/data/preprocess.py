import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

from src.utils.expression import (
    get_gene_embeddings,
    get_normalized_gene_expression,
    load_sample_expression,
)
from src.utils.genome import (
    get_chromosome_lengths,
    get_cds_coord,
    get_chromosome_valid_genes,
    parse_gff,
)


def preprocess(
    gff: str | Path,
    fasta: str | Path,
    embed_dir: str | Path,
    rna_dir: str | Path,
    condition_samples_json: str | Path,
    outdir: str | Path,
    window_size: int = 500,
):
    """
    Create a HDF5 file containing an embedding and an expression value for each gene.

    Parameters
    ----------
    gff : str | Path
        Path to the GFF file.
    fasta : str | Path
        Path to the FASTA file.
    embed_dir : str | Path
        Path to the directory containing the embeddings.
    rna_dir : str | Path
        Path to the directory containing the RNA expression data.
    condition_samples_json : str | Path
        Path to a JSON file containing a mapping of sample names to conditions.
    outdir : str | Path
        Path to the directory to save the HDF5 file.
    window_size : int
        Size of the upstream window.
    """
    gff = Path(gff)
    fasta = Path(fasta)
    embed_dir = Path(embed_dir)
    rna_dir = Path(rna_dir)
    condition_samples_json = Path(condition_samples_json)
    outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    with open(condition_samples_json, "r") as f:
        condition_samples = json.load(f)

    samples = [sample for samples in condition_samples.values() for sample in samples]

    annotation = parse_gff(gff)

    chromosome_lengths = get_chromosome_lengths(fasta)
    cds_coords = get_cds_coord(annotation)

    valid_genes = get_chromosome_valid_genes(
        cds_coords, chromosome_lengths, window_size=window_size
    )

    summary = pd.DataFrame()

    sample_expression = load_sample_expression(rna_dir, samples)

    with h5py.File(outdir / "genewise.h5", "w") as h5f:
        for chromosome, length in tqdm(chromosome_lengths.items()):
            valid_cds_coords = [cds_coords[gene] for gene in valid_genes[chromosome]]

            chromosome_embedding = np.load(f"{embed_dir}/{chromosome}.npy").astype(
                np.float16
            )

            assert len(chromosome_embedding) == length, (
                f"Length of {chromosome} embedding is {len(chromosome_embedding)} but should be {length}"
            )

            gene_embeddings = get_gene_embeddings(
                valid_cds_coords, chromosome_embedding, window_size=window_size
            )

            del chromosome_embedding

            gene_expression = get_normalized_gene_expression(
                valid_cds_coords,
                condition_samples,
                sample_expression
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
                }
            )

            summary = pd.concat([summary, chromosome_df], ignore_index=True)

    summary.to_csv(outdir / "summary.csv", index=False)
