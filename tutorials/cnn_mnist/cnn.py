#!/usr/bin/env python3

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CNN(nn.Module):
    """
    Convolutional Neural Network.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


def load_mnist() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset (download if necessary) and split data into training,
        validation, and test sets.

    Args:
        None
    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """
    TRAIN_PCT = 0.8
    # Specify transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # Load training set
    train_valid_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Partition into training/validation
    n_train_examples = int(TRAIN_PCT * len(train_valid_set))
    n_valid_examples = len(train_valid_set) - n_train_examples
    train_set, valid_set = torch.utils.data.random_split(
        train_valid_set, lengths=[n_train_examples, n_valid_examples]
    )
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True, num_workers=2)

    # Load test set
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader


def train(
    train_loader: DataLoader,
    params: Dict[str, float],
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """
    Train CNN on provided data set.

    Args:
        train_loader: DataLoader containing training set
        params: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.9)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.
    """
    # Initialize network
    net = CNN().to(device=device)
    net.train()
    params["lr"] = params.get("lr", 0.001)
    params["momentum"] = params.get("lr", 0.9)
    # Define loss and optimizer
    criterion = nn.NLLLoss(reduction="sum")
    optimizer = optim.SGD(net.parameters(), **params)

    # Train Network
    for inputs, labels in train_loader:
        # move data to proper dtype and device
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return net


def evaluate(
    net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device
) -> float:
    """
    Compute classification ccuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
