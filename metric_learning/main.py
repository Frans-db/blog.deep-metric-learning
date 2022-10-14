import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from typing import Tuple
import imageio
import uuid
import os

from distances import EuclideanDistance
from losses import ContrastiveLoss
from miners import ContrastiveMiner
from networks import LecunConvolutionalNetwork


def get_device() -> torch.device:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', type=str, default='results',
                        help='Name of the directory to store experiments in')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the current experiment. Used to store results')
    parser.add_argument('--dimensionality', type=int, default=2,
                        help='Manifold dimensionality to map the data to')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--test_every', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader number of workers')

    args = parser.parse_args()

    if args.experiment_name == None:
        args.experiment_name = uuid.uuid4()

    return args


def load_data(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    return trainloader, testloader


def create_directores(results_root: str, experiment_name: str):
    if not os.path.isdir(f'./{results_root}'):
        os.mkdir(f'./{results_root}')
    if not os.path.isdir(f'./{results_root}/{experiment_name}'):
        os.mkdir(f'./{results_root}/{experiment_name}')


def main() -> None:
    device = get_device()
    args = handle_arguments()
    create_directores(args.results_root, args.experiment_name)
    results_directory = f'{args.results_root}/{args.experiment_name}'

    trainloader, testloader = load_data(args)

    miner = ContrastiveMiner(dimensionality=args.dimensionality)
    criterion = ContrastiveLoss(distance=EuclideanDistance())
    network = LecunConvolutionalNetwork(
        dimensionality=args.dimensionality).to(device)
    optimizer = optim.Adam(network.parameters())

    iteration = 0
    image_paths = []
    for epoch in range(args.epochs):
        print(f'Epoch [{epoch:2}]')
        for (inputs, labels) in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if iteration % args.test_every == 0:
                print(f'Iteration [{iteration:4}]')
                image_path = f'./{results_directory}/{epoch}_{iteration}.png'
                test_results, test_labels = test(
                    network, testloader, args.dimensionality)
                scatter(test_results, test_labels, image_path)
                image_paths.append(image_path)

            optimizer.zero_grad()
            outputs = network(inputs)
            outputs, labels = miner(outputs, labels)
            loss = criterion(outputs[0], outputs[1], labels[0], labels[1])
            loss.backward()
            optimizer.step()

            iteration += 1

    results, labels = test(network, testloader, args.dimensionality)
    create_gif(image_paths, results_directory)


def test(network: nn.Module, testloader: DataLoader, dimensionality: int) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        all_results = torch.zeros(0, dimensionality)
        all_labels = torch.zeros(0)
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(get_device()), labels.to(get_device())
            outputs = network(inputs)

            all_results = torch.cat((all_results, outputs.detach().cpu()))
            all_labels = torch.cat((all_labels, labels.detach().cpu()))
    return all_results, all_labels


def scatter(results: torch.Tensor, labels: torch.Tensor, image_path: str) -> None:
    for label in torch.unique(labels):
        idx = labels == label
        embeddings = results[idx].transpose(0, 1)
        plt.scatter(embeddings[0], embeddings[1], label=label.item())
    plt.savefig(image_path)
    plt.clf()


def create_gif(image_names: list[str], results_directory: str) -> None:
    images = []
    for filename in image_names:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'./{results_directory}/movie.gif'), images


if __name__ == '__main__':
    main()
