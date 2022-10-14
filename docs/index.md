<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


# Deep Metric Learning

## Introduction
An HD (1920x1080) image has 2073600 pixels. If we completely ignore colour and assume each pixel is grayscale that image will still have a dimensionality of 2 million. I have a hard time imagining 3D things in my head, let alone 4D or even 2 million dimensions. This is where deep metric learning can be of use to us.

Deep metric learning is a Deep Learning method that tries to learn how to best reduce the dimensionality of inputs, from for example 10D down to a much more manageble (and visible) 3D, 2D, or 1D. In this post I will be exploring some of the first algorithms and tehcniques used to implement deep metric learning, and will be applying them to the MNIST handwritten digit dataset.  Each image in the MNIST digit dataset is a grayscale 28x28 pixel image, which means that each image has 784 dimensions. Through deep metric learning I'll be attempting to reduce this to a 2D space.

Throughout each section I'll show the code used to implement each part needed to perform deep metric learning, which will hopefully serve as start off point for anyone interested in implementing this themselves.

First I will cover the more basic deep metric learning technique, called contrastive learning. After that I'll dive into triplet learning and show some of the different results.

## Contrastive Learning

I'll be following along with [1] for my implementation of contrastive learning.

Contrastive learning works on the premise that points that are similar should be placed close together in the output manifold, and ponints that are different should be placed far apart. It does this by comparing (contrasting) two points against each other, and calculating the loss.

First of all each point $X_i$ is put through the network, 

### Contrastive Loss
The contrastive loss consists of two parts, a similar and a dissimilar part.

$$

$$


$$
L(W, Y) = (1 - Y) \frac{1}{2}(D_W)^2 + (Y) \frac{1}{2}(\max(0, m - D_W))^2
$$

```python
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
```

```python
class EuclideanDistance(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # sqrt( sum_i (x_i - y_i)^2)
        return (x - y).square().sum(dim=-1).sqrt()
```


### Contrastive Miner

```python
class ContrastiveMiner(nn.Module):
    def __init__(self, dimensionality: int = 2) -> None:
        super().__init__()
        self.dimensionality = dimensionality

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transform data from
        # 0 1 2 3 4 5 6 7 8 9 to [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
        # to create pairs of data
        embeddings = embeddings.reshape(2, embeddings.shape[0] // 2, self.dimensionality)
        labels = labels.reshape(2, labels.shape[0] // 2)
        return embeddings, labels
```

## Triplet Learning

### Triplet Loss

```python
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
```

### Triplet Miner

## Network

```python
class LecunConvolutionalNetwork(nn.Module):
    """
    Convolutional Network as described in
    Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
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
```

## Results

## Citations
1. R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality reduction by learning an invariant map-
ping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition
(CVPR’06), volume 2, pages 1735–1742, 2006.
2. Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face
recognition and clustering. CoRR, abs/1503.03832, 2015.