import torch
import torch.nn as nn


class EuclideanDistance(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # sqrt( sum_i (x_i - y_i)^2)
        return (x - y).square().sum(dim=-1).sqrt()
