from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


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

def extract_snippets_from_upstream(
    df, 
    cds_coords, 
    fasta_path, 
    rename_map, 
    upstream_window=500, 
    motif_window=20
):
    """
    Adds a 'snippet' column to the DataFrame containing sequences around max_position
    from the upstream region of each gene.

    Args:
        df (pd.DataFrame): DataFrame with columns 'gene' and 'max_position'.
        cds_coords (dict): Dict mapping gene -> {'chromosome', 'strand', 'coordinates'}.
        fasta_path (str): Path to the genome FASTA file.
        rename_map (dict): Mapping from FASTA record IDs to chromosome names.
        upstream_window (int): Length of upstream region to extract.
        motif_window (int): Half-width of the motif snippet (total length = 2*motif_window).

    Returns:
        pd.DataFrame: Modified DataFrame with added 'snippet' column.
    """
    for gene in df.gene.unique():
        if gene not in cds_coords:
            continue

        upstream_window_coordinates = get_upstream_window_coordinates(cds_coords[gene], upstream_window)
        
        gene_data = {
            'chromosome': cds_coords[gene]["chromosome"],
            'strand': cds_coords[gene]["strand"],
            'coordinates': [(upstream_window_coordinates[0], upstream_window_coordinates[1])]
        }

        seq = get_gene_sequence(gene_data, fasta_path, rename_map=rename_map)

        for idx, row in df[df.gene == gene].iterrows():
            snippet = get_snippet_around_position(seq, row["max_position"], motif_window)
            df.at[idx, "snippet"] = snippet

    return df

def parse_gff_old(gff_path):
    # Get coordinates of the coding sequences
    # The format is {gene_id: {
    #     "chromosome": "chrI",
    #     "strand": "+",
    #     "coordinates": [(start, end), (start, end), ...]
    # }}
    cds_coords = defaultdict(dict)
    with open(gff_path) as gff_file:
        for line in gff_file:
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) < 9:
                continue
            if columns[2] != "CDS":
                continue
            attrs = {kv.split("=")[0]: kv.split("=")[1] for kv in columns[8].split(";")}
            gene = attrs.get("Name")[:-4]
            if gene is None:
                continue
            start, end = int(columns[3]) - 1, int(
                columns[4]
            )  # 1-based to 0-based, end excluded
            chrom = columns[0]
            if chrom == "chrmt":
                chrom = "chrM"
            cds_coords[gene]["chromosome"] = chrom
            cds_coords[gene]["strand"] = columns[6]
            if "coordinates" not in cds_coords[gene]:
                cds_coords[gene]["coordinates"] = []
            cds_coords[gene]["coordinates"].append((start, end))
    return cds_coords

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

def get_snippet_around_position(seq, position, window):
    start = max(position - window, 0)
    end = min(position + window, len(seq))
    return seq[start:end]

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

def get_gene_sequence(gene_info, fasta_path, rename_map=None):
    """
    Reconstructs gene sequence from FASTA file based on coordinates and strand info.

    Args:
        gene_info (dict): Dictionary with structure like:
                          {
                              'chromosome': 'chrI',
                              'strand': '+',
                              'coordinates': [(start1, end1), (start2, end2), ...]
                          }
        fasta_path (str): Path to the FASTA file.
        rename_map (dict): Optional mapping from FASTA chromosome IDs to standard names.

    Returns:
        str: Nucleotide sequence of the gene.
    """
    # Load the fasta into a dictionary of {chromosome_name: sequence}
    fasta_records = {
        (rename_map.get(record.id, record.id) if rename_map else record.id): record.seq
        for record in SeqIO.parse(fasta_path, "fasta")
    }

    chrom = gene_info["chromosome"]
    strand = gene_info["strand"]
    coords = gene_info["coordinates"]

    if chrom not in fasta_records:
        raise ValueError(f"Chromosome {chrom} not found in FASTA.")

    sequence = fasta_records[chrom]

    # Extract and concatenate all regions
    gene_seq = ''.join([str(sequence[start:end]) for start, end in coords])

    # Reverse complement if on the negative strand
    if strand == "-":
        gene_seq = str(Seq(gene_seq).reverse_complement())

    return gene_seq
