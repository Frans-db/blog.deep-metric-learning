# Deep Metric Learning

## Introduction
An HD (1920x1080) image has 2073600 pixels. If we completely ignore colour and assume each pixel is grayscale that image will still have a dimensionality of 2 million. I have a hard time imagining 3D things in my head, let alone 4D or even 2 million dimensions. This is where deep metric learning can be of use to us.

Deep metric learning is a Deep Learning method that tries to learn how to best reduce the dimensionality of inputs, from for example 10D down to a much more manageble (and visible) 3D, 2D, or 1D. In this post I will be exploring some of the first algorithms and tehcniques used to implement deep metric learning, and will be applying them to the MNIST handwritten digit dataset.  Each image in the MNIST digit dataset is a grayscale 28x28 pixel image, which means that each image has 784 dimensions. Through deep metric learning I'll be attempting to reduce this to a 2D space.

Throughout each section I'll show the code used to implement each part needed to perform deep metric learning, which will hopefully serve as start off point for anyone interested in implementing this themselves.

First I will cover the more basic deep metric learning technique, called contrastive learning. After that I'll dive into triplet learning and show some of the different results.

## Contrastive Learning

### Contrastive Loss

### Contrastive Miner

## Triplet Learning

### Triplet Loss

### Triplet Miner