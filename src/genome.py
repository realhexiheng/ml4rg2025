from collections import defaultdict
from pathlib import Path
from Bio import SeqIO
from utils import get_upstream_window_coordinates

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


def get_chromosome_lengths(fasta_path: Path):
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


def parse_gff(gff_path):
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
            start, end = (
                int(columns[3]) - 1,
                int(columns[4]),
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
