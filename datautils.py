"""
datautils.py
------------
Simple synthetic dataset used by the DDP training scripts.
Replace the contents of MyTrainDataset with your real dataset when ready.
"""

import torch
from torch.utils.data import Dataset


class MyTrainDataset(Dataset):
    """
    Synthetic dataset that generates random (input, label) pairs.

    Parameters
    ----------
    size      : int   Total number of samples.
    input_dim : int   Dimension of each input vector  (default 20).
    num_class : int   Number of output classes         (default 2).
    """

    def __init__(self, size: int, input_dim: int = 20, num_class: int = 2):
        self.size      = size
        self.input_dim = input_dim
        self.num_class = num_class

        # Generate everything up front so workers don't re-generate per epoch
        self.data   = torch.randn(size, input_dim)
        self.labels = torch.randint(0, num_class, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]
