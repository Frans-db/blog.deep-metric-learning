import torch.nn as nn
import torch


class TripletLoss(nn.Module):
    """
    Triplet Loss as described in
    FaceNet: A Unified Embedding for Face Recognition and Clustering
    https://arxiv.org/abs/1503.03832
    """
    def __init__(self, distance: nn.Module, margin: float = 1.0) -> None:
        super().__init__()
        self.distance = distance
        self.margin = margin

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        # calculate distances of anchor positive pairs
        positive_distances = self.distance(anchors, positives).square()
        # calculate distances of anchor negative pairs
        negative_distances = self.distance(anchors, negatives).square()
        # loss = (d_pos - d_neg + margin)
        return (positive_distances - negative_distances + self.margin).clamp(0).sum()
