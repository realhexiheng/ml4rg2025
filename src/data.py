import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class CustomDataset(Dataset):
    def __init__(
        self,
        chromosomes: List[str],
        chr_genes: Dict[str, List[str]],
        cache_size: int = 100,
    ):
        genes = []
        for chromosome in chromosomes:
            genes.extend(chr_genes[chromosome])
        self.genes = genes
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        gene = self.genes[idx]

        # Check if data is in cache
        if gene in self.cache:
            return self.cache[gene]

        # Load data from disk
        arr = np.load(f"../data/genes/{gene}.npz")
        data = (arr["X"].astype(np.float32), arr["Y"].astype(np.float32))

        # Add to cache if not full
        if len(self.cache) < self.cache_size:
            self.cache[gene] = data

        return data


def get_data_loader(
    chromosomes: List[str],
    chr_genes: Dict[str, List[str]],
    batch_size: int = 1024,
    cache_size: int = 100,
):
    dataset = CustomDataset(chromosomes, chr_genes, cache_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
