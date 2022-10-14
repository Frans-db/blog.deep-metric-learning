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
    def __init__(self, distance: nn.Module, margin: float = 1.0, positive_weight: float = 0.5) -> None:
        super().__init__()
        self.distance = distance
        self.margin = margin
        self.positive_weight = positive_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_labels: torch.Tensor, y_labels: torch.Tensor) -> torch.Tensor:
        equal: torch.Tensor = x_labels == y_labels
        distances: torch.Tensor = self.distance(x, y)

        # Calculate the positive loss $(D_W)^2$
        positive_loss = equal * self.positive_weight * (distances).square()
        # Calculate the negative loss $max(0, m - D_W)^2$
        negative_loss = (~equal) * (1 - self.positive_weight) * \
            (self.margin - distances).clamp(0).square()

        return (positive_loss + negative_loss).sum()

```

### Contrastive Miner

## Triplet Learning

### Triplet Loss

### Triplet Miner

## Citations
1. R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality reduction by learning an invariant map-
ping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition
(CVPR’06), volume 2, pages 1735–1742, 2006.
2. Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face
recognition and clustering. CoRR, abs/1503.03832, 2015.