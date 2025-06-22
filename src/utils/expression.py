from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.genome import get_upstream_window_coordinates


def load_sample_expression(
    rna_dir: str | Path,
    samples: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Load sample expression data from .npz files.
    Args:
        rna_dir (str | Path): Directory containing RNA expression .npz files.
        samples (list[str]): List of sample names to load.
    Returns:
        dict: Dictionary mapping sample names to expression data:
        {
            sample_name: {
                "+": np.ndarray,  # Sense strand expression
                "-": np.ndarray,  # Antisense strand expression
            }
        }
    """

    expression = {
        sample: {
            "+": np.load(f"{Path(rna_dir)}/{sample}.sense_bp1.npz"),
            "-": np.load(f"{Path(rna_dir)}/{sample}.antisense_bp1.npz"),
        }
        for sample in samples
    }

    return expression


def get_gene_count(
    cds_coord: tuple[int],
    sample_expression: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Get total gene expression count for a given CDS coordinate."""
    strand = cds_coord["strand"]
    chrom = cds_coord["chromosome"]

    samples = sorted(list(sample_expression.keys()))

    # Sum exon segments directly to avoid large temporary concatenated arrays
    expression_values = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        total = 0.0

        for start, end in cds_coord["coordinates"]:
            total += sample_expression[sample][strand][chrom][start:end].sum()

        expression_values[i] = total

    return expression_values


def get_gene_length(cds_coord):
    return sum(end - start for start, end in cds_coord["coordinates"])


def get_gene_embeddings(
    cds_coords: list[dict],
    chromosome_embedding: np.ndarray,
    window_size: int = 500,
) -> np.ndarray:
    """Get gene embeddings for a list of CDS coordinates using vectorized extraction.
    Args:
        cds_coords (list[dict]): List of gene CDS coordinates in the format:
            Example: [{coordinates: [(start, end), ...], chromosome: str, strand:
            str},...]
        chromosome_embedding (np.ndarray): Precomputed chromosome embedding with shape
            (chromosome_length, 768).
        window_size (int): Size of the upstream window to extract (default 500).
    Returns:
        np.ndarray: Gene embeddings with shape (num_genes, window_size, 768).
    """

    n_genes = len(cds_coords)
    gene_embeddings = np.zeros(
        (n_genes, window_size, 768), dtype=chromosome_embedding.dtype
    )
    starts = np.empty(n_genes, dtype=int)
    strands = np.empty(n_genes, dtype="U1")

    for i, cds_coord in enumerate(cds_coords):
        start, _, strand = get_upstream_window_coordinates(cds_coord, window_size)
        starts[i] = start
        strands[i] = strand

    # Vectorized extraction
    window = np.arange(window_size)
    indices = starts[:, None] + window[None, :]  # shape: (n_genes, window_size)

    gene_embeddings = chromosome_embedding[indices]  # shape: (n_genes, window_size, 768)

    # reverse strand
    mask = strands == "-"
    gene_embeddings[mask] = gene_embeddings[mask, ::-1, :]

    return gene_embeddings


def calculate_tpm(gene_counts: np.ndarray, gene_lengths: np.ndarray) -> np.ndarray:
    """Calculates Transcripts Per Million (TPM) from raw counts."""
    sample_count_sum = gene_counts.sum(axis=0)
    gene_rpm = (gene_counts / sample_count_sum) * 1e6
    gene_tpm = (gene_rpm.T / gene_lengths).T
    return gene_tpm


def aggregate_tpm_by_condition(
    gene_tpm: np.ndarray,
    samples: list[str],
    condition_samples: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Averages TPM values across replicate samples for each condition."""
    conditions = sorted(list(condition_samples.keys()))
    condition_tpm = np.zeros((gene_tpm.shape[0], len(conditions)))

    for i, condition in enumerate(conditions):
        c_samples = condition_samples[condition]
        sample_indices = [samples.index(sample) for sample in c_samples]
        condition_mean = gene_tpm[:, sample_indices].mean(axis=1)
        condition_tpm[:, i] = condition_mean

    return condition_tpm, conditions


def get_normalized_gene_expression(
    cds_coords: list[dict],
    condition_samples: dict[str, list[str]],
    sample_expression: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Get normalized gene expression for each condition.
    Args:
        cds_coords (list): List of gene CDS coordinates in the format:
            Example: [{coordinates: [(start, end), ...], chromosome: str, strand: str},...]
        condition_samples (dict): Dictionary mapping conditions to sample names:
            Example: {"condition1": ["sample1", "sample2"], "condition2": ["sample3"]}
        sample_expression (dict): Dictionary mapping sample names to expression data:
            Example: {
                "sample1": {"+": {chromosome: np.ndarray, ...}, "-": {chromosome: np.ndarray, ...}},
                "sample2": {"+": {chromosome: np.ndarray, ...}, "-": {chromosome: np.ndarray, ...}},
                ...
            }
    Returns:
        np.ndarray: Normalized gene expression matrix with shape (num_genes, num_conditions).
    """
    samples = sorted(list(sample_expression.keys()))

    gene_counts = np.zeros((len(cds_coords), len(samples)))
    gene_lengths = np.zeros(len(cds_coords))

    for i, cds_coord in enumerate(tqdm(cds_coords)):
        gene_counts[i] = get_gene_count(cds_coord, sample_expression)
        gene_lengths[i] = get_gene_length(cds_coord)

    gene_tpm = calculate_tpm(gene_counts, gene_lengths)

    condition_tpm, _ = aggregate_tpm_by_condition(gene_tpm, samples, condition_samples)

    return np.log1p(condition_tpm).astype(np.float16)
