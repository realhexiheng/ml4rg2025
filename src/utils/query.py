from pandas import DataFrame
from pybiomart import Server


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
