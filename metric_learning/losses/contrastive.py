import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss as described in
    Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, distance: nn.Module, margin: float = 1.0, positive_weight: float = 0.5) -> None:
        super().__init__()
        self.distance = distance
        self.margin = margin
        self.positive_weight = positive_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_labels: torch.Tensor, y_labels: torch.Tensor) -> torch.Tensor:
        # compare x labels to y labels to get simmilar and dissimilar pairs
        equal: torch.Tensor = x_labels == y_labels
        # calculate distances between all pairs
        distances: torch.Tensor = self.distance(x, y)

        # Calculate the positive loss d^2
        positive_loss = equal * self.positive_weight * (distances).square()
        # Calculate the negative loss max(0, m - d)^2
        negative_loss = (~equal) * (1 - self.positive_weight) * \
            (self.margin - distances).clamp(0).square()

        return (positive_loss + negative_loss).sum()
