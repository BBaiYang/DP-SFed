import torch.nn as nn
import numpy as np
import torch
from models.forFashionMNIST.cnn import evaluate_accuracy
from torchvision import transforms
import torchvision
from configs import HYPER_PARAMETERS as hp
from torch.utils import data


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, net):
        super(VGG, self).__init__()
        self.net = net
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net(x)
        return x


def make_layers(cfg, size=512, out=10):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [
                conv2d,
                # nn.BatchNorm2d(v),
                nn.ReLU()]
            in_channels = v
    layers += [FlattenLayer()]
    layers += [nn.Linear(size, size)]
    layers += [nn.ReLU()]
    layers += [nn.Linear(size, size)]
    layers += [nn.ReLU()]
    layers += [nn.Linear(size, out)]
    return nn.Sequential(*layers)


def vgg11s():
    return VGG(make_layers([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'], size=128))


def vgg11():
    return VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']))


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train = torchvision.datasets.CIFAR10(root='~/CODES/Dataset',
                                                    train=True, download=True, transform=train_transform)
    cifar_test = torchvision.datasets.CIFAR10(root='~/CODES/Dataset',
                                                   train=False, download=True, transform=test_transform)

    '''hyper parameters'''
    device = hp['device_for_baseline']
    participating_ratio = hp['participating_ratio']
    sampling_pr = hp['sampling_pr']
    lr = hp['lr']
    momentum = hp['momentum']
    weight_decay = hp['weight_decay']
    batch_size = int(participating_ratio * sampling_pr * len(cifar_train))
    global_epochs = hp['global_epochs']
    batches = int(1 / sampling_pr / participating_ratio)

    net = vgg11s().to(device)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_loader = data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(cifar_test, batch_size=batch_size, num_workers=4)
    for epoch in range(global_epochs):
        for batch, (X, y) in enumerate(train_loader):
            y_pred = net(X.to(device))
            train_l = loss(y_pred, y.to(device))
            trainer.zero_grad()
            train_l.backward()
            trainer.step()
            with torch.no_grad():
                test_acc, test_l = evaluate_accuracy(net, test_loader, loss, device)
                print(f"[Round: {epoch * batches + batch + 1: 04}] "
                        f"Test set: Average loss: {test_l:.4f}, Accuracy: {test_acc:.4f}")
