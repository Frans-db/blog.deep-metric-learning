<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']]
    }
  };
</script>



# Deep Metric Learning

## Introduction
An HD (1920x1080) image has 2073600 pixels. If we completely ignore colour and assume each pixel is grayscale that image will still have a dimensionality of 2 million. I have a hard time imagining 3D things in my head, let alone 4D or even 2 million dimensions. This is where deep metric learning can be of use to us.

Deep metric learning is a Deep Learning method that tries to learn how to best reduce the dimensionality of inputs, from for example 10D down to a much more manageble (and visible) 3D, 2D, or 1D. In this post I will be exploring some of the first algorithms and tehcniques used to implement deep metric learning, and will be applying them to the MNIST handwritten digit dataset.  Each image in the MNIST digit dataset is a grayscale 28x28 pixel image, which means that each image has 784 dimensions. Through deep metric learning I'll be attempting to reduce this to a 2D space.

Throughout each section I'll show the code used to implement each part needed to perform deep metric learning, which will hopefully serve as start off point for anyone interested in implementing this themselves.

First I will cover the basic idea and notation, then I will cover 2 deep metric learning techniques called contrastive learning and triplet learning. Finally I'll show some results and ideas for future work.

## Basics
The basic idea behind deep metric learning is to reduce the dimensionality of an input in such a way that similar samples will be close together in the output space. If we have neural network $G_W(x)$, where $W$ are the parameters, then the goal for two items of the same class is for the distance
$$
d(G_W(x_1), d_W(x_2))
$$
to be small. For two items of dissimilar items we instead want this distance to be large. For this project I'll be using the euclidean distance
$$
\sum_{i=1}^n (x_i - y_i)^2
$$

```python
class EuclideanDistance(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # sqrt( sum_i (x_i - y_i)^2)
        return (x - y).square().sum(dim=-1).sqrt()
```

## Contrastive Learning

I'll be following along with [1] for my implementation of contrastive learning.

Contrastive learning works on the premise that points that are similar should be placed close together in the output manifold, and points that are different should be placed far apart. It does this by comparing (contrasting) two points against each other, and calculating the loss.

### Contrastive Loss
The contrastive loss consists of two parts, a similar and a dissimilar part. The similar part increases when two samples that have the same class are far away from each other.

$$
d(G_W(x_1), G_W(x_2))^2
$$

The dissimilar part increases when two samples with different classes are closer to each other than some margin $m$. This part is needed because with only the similar loss the output of the network would collapse by embedding all inputs to the same point.

$$
max(0, m - d(G_W(x_1), G_W(x_2)))^2
$$

Combining both losses we get

$$
L(W, Y) = (Y) (D_W)^2 + (Y - 1) \max(0, m - D_W)^2
$$

Where $D_W$ is the distance, and Y is a vector which is 1 for similar pairs, and 0 for dissimilar pairs.

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