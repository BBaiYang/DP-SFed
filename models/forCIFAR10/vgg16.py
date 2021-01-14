import torch.nn as nn
from forFashionMNIST.cnn import evaluate_accuracy
import torchvision
from torchvision import transforms
from configs import HYPER_PARAMETERS as hp
import torch
from torch.utils import data
from utils import FED_LOG as fed_log


cfg = {
    'VGG11': (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'),
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class VGG16(nn.Module):
    def __init__(self, config=cfg['VGG16']):
        super(VGG16, self).__init__()
        self.config = config
        # self.features = self._make_layers(self.config)
        # self.flatten = FlattenLayer()
        # self.classifier = nn.Linear(512, 10)
        self.model = self._make_layers(self.config)

    def forward(self, x):
        # out = self.features(x)
        # out = self.flatten(out)
        # out = self.classifier(out)
        out = self.model(x)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [FlattenLayer()]
        layers += [nn.Linear(512, 10)]
        return nn.Sequential(*layers)


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
    sampling_pr = hp['sampling_pr']
    lr = hp['lr']
    momentum = hp['momentum']
    batch_size = int(sampling_pr * len(cifar_train))
    global_epochs = hp['global_epochs']
    batches = int(1 / sampling_pr)

    net = VGG16().to(device)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

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
                fed_log(f"[Round: {epoch * batches + batch + 1: 04}] "
                        f"Test set: Average loss: {test_l:.4f}, Accuracy: {test_acc:.4f}")

    # train_dataloaders, local_sample_sizes, test_dataloader = get_data_loaders()
    # batches = len(train_dataloaders[0])
    # for epoch in range(global_epochs):
    #     for batch, (item0, item1, item2, item3, item4, item5, item6, item7, item8, item9) in \
    #             enumerate(zip(train_dataloaders[0], train_dataloaders[1], train_dataloaders[2], train_dataloaders[3],
    #                           train_dataloaders[4], train_dataloaders[5], train_dataloaders[6], train_dataloaders[7],
    #                           train_dataloaders[8], train_dataloaders[9])):
    #         X = torch.cat((item0[0], item1[0], item2[0], item3[0], item4[0],
    #                       item5[0], item6[0], item7[0], item8[0], item9[0]))
    #         y = torch.cat((item0[1], item1[1], item2[1], item3[1], item4[1],
    #                       item5[1], item6[1], item7[1], item8[1], item9[1]))
    #         # for i in range(0, 2500, 250):
    #         #     print(y[i], y[i+1], end=' ')
    #         # print('')
    #         y_pred = net(X.to(device))
    #         train_l = loss(y_pred, y.to(device))
    #         trainer.zero_grad()
    #         train_l.backward()
    #         trainer.step()
    #         with torch.no_grad():
    #             test_acc, test_l = evaluate_accuracy(net, test_dataloader, loss, device)
    #             fed_log(f"[Round: {epoch * batches + batch + 1: 04}] "
    #                     f"Test set: Average loss: {test_l:.4f}, Accuracy: {test_acc:.4f}")
