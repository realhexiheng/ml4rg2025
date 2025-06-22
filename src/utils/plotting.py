import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow


def plot_region_with_signals(
    cds_coords, data_tracks, chromosome, region_start, region_end
):
    """
    Plot one or more numerical signal tracks across a genomic interval, with gene/CDS boxes
    and strand arrows below. Gene names are drawn on top of each exon box (rather than once per gene).

    Parameters
    ----------
    cds_coords : dict
        Maps each gene name to a dict with keys:
            • 'chromosome' : string (e.g. 'chrIV')
            • 'strand'     : '+' or '-'
            • 'coordinates': list of (start, end) tuples (1-based genomic coordinates)
    data_tracks : dict
        Maps each track label (e.g. a filename or condition name) to a dict mapping
        chromosome name → 1D numpy array of that chromosome's signal values.
        Each array must be length = chromosome length, indexed 0→position 1, etc.
    chromosome : str
        Name of the chromosome to plot (must match cds_coords entries, e.g. 'chrIV').
    region_start : int
        1-based genomic start coordinate of the window.
    region_end : int
        1-based genomic end coordinate of the window.
    """
    # 1) Collect all genes on this chromosome that overlap [region_start, region_end]
    genes_in_region = []
    for gene, info in cds_coords.items():
        if info["chromosome"] != chromosome:
            continue
        for start, end in info["coordinates"]:
            if end >= region_start and start <= region_end:
                genes_in_region.append((gene, info))
                break

    # 2) Number of signal tracks
    num_tracks = len(data_tracks)

    # 3) How many genes we’ll need to display
    num_genes = len(genes_in_region)

    # 4) Calculate figure height
    signal_height = 1.5
    gene_panel_height = max(1.0, num_genes * 0.5 + 0.5)
    total_height = num_tracks * signal_height + gene_panel_height

    fig, axes = plt.subplots(
        nrows=num_tracks + 1, ncols=1, sharex=True, figsize=(10, total_height)
    )
    # If only one signal track, axes is a length-2 array; otherwise length = num_tracks+1
    if num_tracks == 1:
        signal_axes = [axes[0]]
        gene_ax = axes[1]
    else:
        signal_axes = axes[:-1]
        gene_ax = axes[-1]

    # 5) Plot each signal track (with grid lines)
    x = np.arange(region_start, region_end + 1)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, chrom_dict) in enumerate(data_tracks.items()):
        ax = signal_axes[idx]
        array = chrom_dict.get(chromosome)
        if array is None:
            raise ValueError(
                f"No data for chromosome '{chromosome}' in track '{label}'."
            )
        chrom_len = array.shape[0]
        if region_end > chrom_len:
            raise ValueError(
                f"Region end ({region_end}) exceeds length of '{label}' (len={chrom_len})."
            )
        y = array[region_start - 1 : region_end]
        c = color_cycle[idx % len(color_cycle)]
        ax.plot(x, y, color=c, linewidth=1)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xlim(region_start, region_end)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 6) Draw gene/CDS panel (with grid)
    gene_ax.hlines(
        y=num_genes + 1, xmin=region_start, xmax=region_end, color="black", linewidth=2
    )
    gene_ax.text(
        (region_start + region_end) / 2,
        num_genes + 1.2,
        f"{chromosome}: {region_start}-{region_end}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    gene_ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 7) For each gene, draw all exons and place the name on each exon box
    y_positions = list(range(num_genes, 0, -1))
    for y_pos, (gene, info) in zip(y_positions, genes_in_region):
        strand = info["strand"]
        coords = info["coordinates"]

        # 7a) Draw each exon/CDS rectangle, label it with gene name
        for start, end in coords:
            if end < region_start or start > region_end:
                continue
            rect_start = max(start, region_start)
            rect_end = min(end, region_end)
            rect_width = rect_end - rect_start

            # Draw the exon box
            exon_rect = Rectangle(
                (rect_start, y_pos - 0.2),
                rect_width,
                0.4,
                facecolor="skyblue",
                edgecolor="black",
            )
            gene_ax.add_patch(exon_rect)

            # Compute center of this exon for labeling
            exon_center = rect_start + rect_width / 2
            gene_ax.text(
                exon_center,
                y_pos,
                gene,
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )

        # 7b) Draw the strand arrow once (covering entire gene span)
        gene_start = min(s for (s, e) in coords)
        gene_end = max(e for (s, e) in coords)
        arrow_start = max(gene_start, region_start)
        arrow_end = min(gene_end, region_end)
        arrow_length = min(arrow_end - arrow_start, 500)

        if strand == "+":
            arr = FancyArrow(
                x=arrow_start,
                y=y_pos,
                dx=arrow_length,
                dy=0,
                length_includes_head=True,
                head_width=0.2,
                head_length=50,
                color="black",
            )
        else:
            arr = FancyArrow(
                x=arrow_end,
                y=y_pos,
                dx=-arrow_length,
                dy=0,
                length_includes_head=True,
                head_width=0.2,
                head_length=50,
                color="black",
            )
        gene_ax.add_patch(arr)

    # 8) Final formatting of gene_ax
    gene_ax.set_ylim(0, num_genes + 2)
    gene_ax.set_xlim(region_start, region_end)
    gene_ax.set_yticks([])
    gene_ax.set_xlabel("Genomic Position", fontsize=8)
    gene_ax.tick_params(axis="x", labelsize=6)
    gene_ax.spines["top"].set_visible(False)
    gene_ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
