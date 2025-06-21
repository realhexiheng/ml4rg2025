import numpy as np
from tqdm import tqdm


def get_upstream_window_coordinates(cds_coord, window_size):
    strand = cds_coord["strand"]
    coordinates = cds_coord["coordinates"]
    cds_start = coordinates[0][0] if strand == "+" else coordinates[-1][1]
    window_direction = -1 if strand == "+" else 1
    window_start = cds_start + window_size * window_direction

    start = min(window_start, cds_start)
    end = max(window_start, cds_start)

    return start, end, strand


def get_upstream_embedding(cds_coord, chr_embedding, window_size=500):
    start, end, strand = get_upstream_window_coordinates(cds_coord, window_size)

    gene_slize = chr_embedding[start:end]

    if strand == "-":
        gene_slize = gene_slize[::-1]

    return gene_slize


def get_gene_count(cds_coord, sample_expression):
    strand = cds_coord["strand"]
    chrom = cds_coord["chromosome"]

    samples = sorted(list(sample_expression.keys()))

    expression_values = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        expression_values[i] = np.concatenate(
            [sample_expression[sample][strand][chrom][start:end] for start, end in cds_coord["coordinates"]]
        ).sum()

    return expression_values


def get_gene_length(cds_coord):
    return sum(end - start for start, end in cds_coord["coordinates"])


def get_gene_embeddings(cds_coords, chromosome_embedding, window_size=500):
    gene_embeddings = np.zeros((len(cds_coords), window_size, 768))
    for i, cds_coord in enumerate(tqdm(cds_coords)):
        gene_embeddings[i] = get_upstream_embedding(
            cds_coord, chromosome_embedding, window_size
        )
    return gene_embeddings


def get_normalized_gene_expression(cds_coords, condition_samples, sample_expression):
    samples = sorted([s for c_s in condition_samples.values() for s in c_s])

    gene_counts = np.zeros((len(cds_coords), len(samples)))
    gene_lengths = np.zeros(len(cds_coords))
    for i, cds_coord in enumerate(tqdm(cds_coords)):
        gene_counts[i] = get_gene_count(cds_coord, sample_expression)
        gene_lengths[i] = get_gene_length(cds_coord)

    sample_count_sum = gene_counts.sum(axis=0)
    gene_rpm = (gene_counts / sample_count_sum) * 1e6
    gene_tpm = (gene_rpm.T / gene_lengths).T

    conditions = sorted(list(condition_samples.keys()))
    condition_counts = np.zeros((len(cds_coords), len(conditions)))
    for i, condition in enumerate(conditions):
        c_samples = condition_samples[condition]
        sample_indices = [samples.index(sample) for sample in c_samples]
        condition_mean = gene_tpm[:, sample_indices].mean(axis=1)
        condition_counts[:, i] = condition_mean

    return np.log1p(condition_counts).astype(np.float16)
