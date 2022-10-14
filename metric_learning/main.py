import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse

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

    data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # basic contrastive learning setup:
    # - argument parsing
    # - data loading
    # - train loop
    # - (inner) test loop
    # ^ try to combine these 2?
    # - loss graph
    # - result graph
    # later: gif maker

    miner = ContrastiveMiner(dimensionality=args.dimensionality)
    criterion = ContrastiveLoss(distance=EuclideanDistance())
    network = LecunConvolutionalNetwork(dimensionality=args.dimensionality)

    optimizer = optim.Adam(network.parameters())

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
    return


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(
        f'{classes[labels[j]]:5s}' for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    main()
