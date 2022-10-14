import torch
import torch.nn as nn
from typing import Tuple


class ContrastiveMiner(nn.Module):
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transform data from
        # 1 2 3 4 5 6 7 8 9 10
        # to
        # (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)
        embeddings = embeddings.reshape(2, embeddings.shape[0] // 2, 2)
        labels = labels.reshape(2, labels.shape[0] // 2)
        return embeddings, labels
