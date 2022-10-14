import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from distances import EuclideanDistance
from losses import ContrastiveLoss
from miners import ContrastiveMiner
from networks import LecunConvolutionalNetwork


def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensionality', type=int, default=2,
                        help='Manifold dimensionality to map the data to')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader number of workers')

    args = parser.parse_args()

    return args


def load_data(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    return trainloader, testloader


def main():
    args = handle_arguments()
    trainloader, testloader = load_data(args)

    miner = ContrastiveMiner(dimensionality=args.dimensionality)
    criterion = ContrastiveLoss(distance=EuclideanDistance())
    network = LecunConvolutionalNetwork(dimensionality=args.dimensionality)

    optimizer = optim.Adam(network.parameters())

    losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        for (inputs, labels) in trainloader:
            optimizer.zero_grad()
            outputs = network(inputs)
            outputs, labels = miner(outputs, labels)
            loss = criterion(outputs[0], outputs[1], labels[0], labels[1])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'[{epoch:2}] loss: {epoch_loss:.2f}')
        losses.append(epoch_loss)

    all_results = torch.zeros(0, args.dimensionality)
    all_labels = torch.zeros(0)
    for (inputs, labels) in testloader:
        outputs = network(inputs)

        all_results = torch.cat((all_results, outputs))
        all_labels = torch.cat((all_labels, labels))
    all_results, all_labels = all_results.detach(), all_labels.detach()
    for label in torch.unique(all_labels):
        idx = all_labels == label
        embeddings = all_results[idx].transpose(0, 1)
        plt.scatter(embeddings[0], embeddings[1], label=label.item())
    plt.show()

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
