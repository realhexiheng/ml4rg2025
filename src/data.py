import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List


class CustomDataset(Dataset):
    def __init__(self, chromosomes: List[str], window_size: int = 500):
        x_values = []
        y_values = []
        for chromosome in tqdm(chromosomes):
            x, y = load_chromosome(chromosome, window_size)
            x_values.append(x)
            y_values.append(y)

        self.X = np.concatenate(x_values)
        self.Y = np.concatenate(y_values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
