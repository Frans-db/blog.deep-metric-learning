import torch
import torch.nn as nn
from typing import Tuple


class TripletMiner(nn.Module):
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i = 0
        anchors = []
        positives = []
        negatives = []
        while i < len(labels):
            positive = None
            negative = None

            j = i + 1
            while j < len(labels):
                if labels[j] == labels[i] and positive is None:
                    positive = j
                if labels[j] != labels[i] and negative is None:
                    negative = j
                if positive is not None and negative is not None:
                    anchors.append(embeddings[i])
                    positives.append(embeddings[positive])
                    negatives.append(embeddings[negative])
                    break
                j = j + 1
            i = i + 1

        triplets = torch.stack([
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        ])
        return triplets, None
