import torch
import torch.nn as nn
from typing import Tuple


class ContrastiveMiner(nn.Module):
    def __init__(self, dimensionality: int = 2) -> None:
        super().__init__()
        self.dimensionality = dimensionality

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transform data from
        # 0 1 2 3 4 5 6 7 8 9
        # to
        # [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
        embeddings = embeddings.reshape(
            2, embeddings.shape[0] // 2, self.dimensionality)
        labels = labels.reshape(2, labels.shape[0] // 2)
        return embeddings, labels
