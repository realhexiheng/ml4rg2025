from pandas import DataFrame
from pybiomart import Server
from collections import defaultdict


def query_paralogs(genes: list) -> DataFrame:
    """
    Retrieves paralogous genes for a list of S. cerevisiae gene IDs using BioMart.

    Args:
        genes (list): A list of S. cerevisiae gene IDs (e.g., 'YML028W', 'YLR042W').

    Returns:
        pandas.DataFrame: A DataFrame with query gene IDs and their paralog information.
    """
    server = Server(host="http://www.ensembl.org")

    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["scerevisiae_gene_ensembl"]

    attributes = [
        "ensembl_gene_id",
        "scerevisiae_paralog_ensembl_gene",
        "scerevisiae_paralog_chromosome",
        "scerevisiae_paralog_orthology_type",
        "scerevisiae_paralog_perc_id",
    ]

    print(f"Querying BioMart for paralogs of {len(genes)} genes...")

    results = dataset.query(attributes=attributes, use_attr_names=True).rename(
        columns={
            "ensembl_gene_id": "gene",
            "scerevisiae_paralog_ensembl_gene": "paralog_ensembl_id",
            "scerevisiae_paralog_chromosome": "paralog_chromosome",
            "scerevisiae_paralog_orthology_type": "orthology_type",
            "scerevisiae_paralog_perc_id": "percent_identity",
        }
    )

    return results[results["gene"].isin(genes)].reset_index(drop=True)


def create_paralog_clusters(genes: list[str], paralogs: DataFrame) -> dict[str, str]:
    """
    Create paralog groups using direct mapping from reciprocal relationships.

    Args:
        genes (list[str]): List of all valid genes
        paralogs (DataFrame): DataFrame containing paralog information
    Returns:
        dict: Gene to group_id mapping
    """
    cluster = defaultdict(set)

    paralogs = paralogs[paralogs["paralog_ensembl_id"].notna()]
    paralogs = paralogs[paralogs["paralog_ensembl_id"].isin(genes)]

    for _, row in paralogs.iterrows():
        gene1, gene2 = row["gene"], row["paralog_ensembl_id"]

        cluster[gene1].add(gene1)
        cluster[gene1].add(gene2)
        cluster[gene2].add(gene2)
        cluster[gene2].add(gene1)

    mapping = {}
    for representative, genes_in_cluster in cluster.items():
        for gene in genes_in_cluster:
            mapping[gene] = representative

    standalone = set(genes) - set(mapping.keys())
    mapping.update({gene: gene for gene in standalone})

    return mapping
