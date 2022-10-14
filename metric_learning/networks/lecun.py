# Network from Dimensionality Reduction by Learning an Invariant Mapping
# http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

import torch.nn as nn
import torch.nn.functional as F
import torch


class LecunConvolutionalNetwork(nn.Module):
    def __init__(self, dimensionality: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 15, 6)
        self.pool1 = nn.AvgPool2d(3)
        self.conv2 = nn.Conv2d(15, 30, 9)
        self.fc1 = nn.Linear(30, dimensionality)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
