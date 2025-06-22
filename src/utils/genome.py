from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO

FASTA_CHROMOSOME_RENAME_MAP = {
    "ref|NC_001133|": "chrI",
    "ref|NC_001134|": "chrII",
    "ref|NC_001135|": "chrIII",
    "ref|NC_001136|": "chrIV",
    "ref|NC_001137|": "chrV",
    "ref|NC_001138|": "chrVI",
    "ref|NC_001139|": "chrVII",
    "ref|NC_001140|": "chrVIII",
    "ref|NC_001141|": "chrIX",
    "ref|NC_001142|": "chrX",
    "ref|NC_001143|": "chrXI",
    "ref|NC_001144|": "chrXII",
    "ref|NC_001145|": "chrXIII",
    "ref|NC_001146|": "chrXIV",
    "ref|NC_001147|": "chrXV",
    "ref|NC_001148|": "chrXVI",
    "ref|NC_001224|": "chrM",
}


def parse_gff(gff: str | Path) -> pd.DataFrame:
    """
    Load and process GFF File, ignoring FASTA sequence at the end.
    Args:
        gff (str | Path): Path to the GFF file.

    Returns:
        pd.DataFrame: Processed GFF data with standardized chromosome names
    """
    with open(gff, "r") as f:
        lines = f.readlines()

    # Find where the FASTA sequence starts (marked by '###') and exclude it
    try:
        fasta_start_index = [i for i, line in enumerate(lines) if line.startswith("###")][
            0
        ]
        lines = "".join(lines[:fasta_start_index])
    except IndexError:
        lines = "".join(lines)

    import io

    columns = [
        "chromosome",
        "feature",
        "start",
        "end",
        "strand",
        "attributes",
    ]

    annotation = pd.read_csv(
        io.StringIO(lines),
        sep="\t",
        comment="#",
        usecols=[0, 2, 3, 4, 6, 8],
        names=columns,
        dtype={
            "start": int,
            "end": int,
        },
        na_filter=False,
    )

    annotation["chromosome"] = annotation["chromosome"].replace({"chrmt": "chrM"})

    return annotation


def get_upstream_window_coordinates(
    cds_coord: dict,
    window_size: int = 500,
) -> tuple[int, int, str]:
    """
    Get upstream promoter region coordinates for a gene.

    Args:
        cds_coord: Gene coordinate info with chromosome, strand, coordinates
        window_size: Size of upstream window (default 500bp)

    Returns:
        Tuple[start, end, strand] for promoter region
    """
    strand = cds_coord["strand"]
    coordinates = cds_coord["coordinates"]
    cds_start = coordinates[0][0] if strand == "+" else coordinates[-1][1]
    window_direction = -1 if strand == "+" else 1
    window_start = cds_start + window_size * window_direction

    start = min(window_start, cds_start)
    end = max(window_start, cds_start)

    return start, end, strand


def get_chromosome_lengths(fasta_path: Path) -> dict[str, int]:
    """Get lengths of chromosomes from a FASTA file
    Args:
        fasta_path (Path): Path to the FASTA file containing chromosome sequences.
    Returns:
        dict: Dictionary mapping chromosome names to their lengths:
        {chromosome: length}
    """

    chromosome_lengths = {}
    for seq in SeqIO.parse(fasta_path, "fasta"):
        chromosome_lengths[FASTA_CHROMOSOME_RENAME_MAP[seq.id]] = len(seq)
    return chromosome_lengths


def get_chromosome_valid_genes(
    cds_coords: dict, chromosome_lengths: dict, window_size: int = 500
):
    chromosome_gene_map = defaultdict(list)
    for gene, cds_coord in cds_coords.items():
        chromosome = cds_coord["chromosome"]
        upstream_window_coordinates = get_upstream_window_coordinates(
            cds_coord, window_size
        )
        if (
            upstream_window_coordinates[0] < 0
            or upstream_window_coordinates[1] > chromosome_lengths[chromosome]
        ):
            continue
        chromosome_gene_map[chromosome].append(gene)
    return chromosome_gene_map


def get_cds_coord(annotation: pd.DataFrame) -> dict[str, dict]:
    """
    Extract CDS coordinates from GFF annotation.

    Args:
        annotation (pd.DataFrame): Processed GFF in DataFrame format.

    Returns:
        dict: DataFrame containing CDS coordinates in the format:
            {gene_id :{
                "chromosome": str,
                "strand": str,
                "coordinates": [(start, end), (start, end), ...]
            }}
    """
    cds_df = annotation[annotation["feature"] == "CDS"].copy()

    cds_df["gene_id"] = cds_df["attributes"].str.extract(r"Name=([^;]+)_CDS")[0]

    cds_coords = defaultdict(lambda: {"coordinates": []})

    for gene_id, group in cds_df.groupby("gene_id"):
        first_row = group.iloc[0]
        cds_coords[gene_id]["chromosome"] = first_row["chromosome"]
        cds_coords[gene_id]["strand"] = first_row["strand"]
        coords = sorted(zip(group["start"], group["end"]))
        cds_coords[gene_id]["coordinates"] = coords

    return dict(cds_coords)
