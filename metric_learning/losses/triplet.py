import torch.nn as nn
import torch


class TripletLoss(nn.Module):
    def __init__(self, distance: nn.Module, margin: float = 1.0, positive_weight: float = 0.5) -> None:
        super().__init__()
        self.distance = distance
        self.margin = margin
        self.positive_weight = positive_weight

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        positive_distances = self.distance(anchors, positives).square()
        negative_distances = self.distance(anchors, negatives).square()

        return (positive_distances - negative_distances + self.margin).clamp(0).sum()
