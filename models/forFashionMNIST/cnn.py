import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils import data
from configs import HYPER_PARAMETERS as hyper_parameters
from utils import FED_LOG as fed_log


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    cmp = (y_hat.argmax(dim=1).type(y.dtype) == y).float().sum().item()
    return cmp


def evaluate_accuracy(net, data_iter, loss, device):
    net.eval()
    metric = Accumulator(3)
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        y_pred = net(X)
        test_acc = accuracy(y_pred, y)
        test_l = loss(y_pred, y)
        metric.add(test_acc, test_l * len(y), len(y))
    net.train()
    return metric[0] / metric[2], metric[1] / metric[2]


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = FlattenLayer()
        self.fc1 = nn.Linear(1024, 512)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.maxpool1(self.act1(self.conv1(x)))
        out = self.maxpool2(self.act2(self.conv2(out)))
        out = self.flatten(out)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = torchvision.datasets.FashionMNIST(root='~/CODES/Dataset',
                                                    train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/CODES/Dataset',
                                                   train=False, download=True, transform=transform)

    '''hyper parameters'''
    device = hyper_parameters['device_for_baseline']
    sampling_pr = hyper_parameters['sampling_pr']
    lr = hyper_parameters['lr']
    momentum = hyper_parameters['momentum']
    batch_size = int(sampling_pr * len(mnist_train))
    global_epoch = hyper_parameters['communication_rounds']
    batches = int(1 / sampling_pr)

    net = CNN().to(device)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, num_workers=4)
    for epoch in range(global_epoch):
        for batch, (X, y) in enumerate(train_loader):
            y_pred = net(X.to(device))
            train_l = loss(y_pred, y.to(device))
            trainer.zero_grad()
            train_l.backward()
            trainer.step()
            with torch.no_grad():
                test_acc, test_l = evaluate_accuracy(net, test_loader, loss, device)
                fed_log(f"[Round: {epoch * batches + batch + 1: 04}] "
                        f"Test set: Average loss: {test_l:.4f}, Accuracy: {test_acc:.4f}")
