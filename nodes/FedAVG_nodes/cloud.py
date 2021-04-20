import torch
import torch.nn as nn
from models.model_factory import model_factory
from configs import HYPER_PARAMETERS as hp
from configs import FedAVG_model_path, FedAVG_aggregated_model_path
import random


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
    cmp = (y_hat.argmax(dim=1).type(y.dtype) == y).sum().float()
    return cmp


def evaluate_accuracy(model, data_iter):
    model.eval()
    metric = Accumulator(3)
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        test_acc = accuracy(y_pred, y)
        test_l = loss(y_pred, y)
        metric.add(test_acc, test_l * len(y), len(y))
    model.train()
    return metric[0] / metric[2], metric[1] / metric[2]


class Cloud:
    def __init__(self, clients, dataloader):
        self.model = model_factory(data_set_name, model_name).to(device)
        print(self.model)
        self._save_model()
        self.clients = clients
        self.total_size = 0
        for client in self.clients:
            self.total_size += client.data_size
        self.test_loader = dataloader
        self.participating_clients = None

    def initialize(self):
        self.total_size = 0
        self.participating_clients = random.sample(self.clients, int(participating_ratio * len(self.clients)))
        for client in self.participating_clients:
            self.total_size += client.data_size

    def aggregate(self):
        aggregated_client_model = {}
        for k, client in enumerate(self.participating_clients):
            weight = client.data_size / self.total_size
            for name, param in client.model.state_dict().items():
                if k == 0:
                    aggregated_client_model[name] = param.data * weight
                else:
                    aggregated_client_model[name] += param.data * weight
        self.model.load_state_dict(aggregated_client_model)
        self._save_params()

    def validate(self):
        with torch.no_grad():
            test_acc, test_l = evaluate_accuracy(self.model, self.test_loader)
        return test_acc, test_l

    def _save_model(self):
        print('initialize the model...')
        torch.save(self.model, FedAVG_model_path)

    def _save_params(self):
        torch.save(self.model.state_dict(), FedAVG_aggregated_model_path)


device = hp['device']
loss = nn.CrossEntropyLoss()
participating_ratio = hp['participating_ratio']
data_set_name = hp['dataset']
model_name = hp['model_name']
