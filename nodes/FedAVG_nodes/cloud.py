"""
Cloud server
"""
import torch
import torch.nn as nn
from models.model_factory import model_factory
from configs import HYPER_PARAMETERS as hp
from configs import DATA_SET_NAME as data_set_name
from configs import MODEL_NAME as model_name
from configs import FedAVG_model_path, FedAVG_aggregated_model_path
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
    cmp = (y_hat.argmax(dim=1).type(y.dtype) == y).sum().float()
    return cmp


def evaluate_accuracy(model, data_iter):
    metric = Accumulator(3)
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        test_acc = accuracy(y_pred, y)
        test_l = loss(y_pred, y)
        metric.add(test_acc, test_l * len(y), len(y))
    return metric[0] / metric[2], metric[1] / metric[2]


class CloudSever:
    def __init__(self, clients, dataloader):
        self.model = model_factory(data_set_name, model_name).to(device)
        self._save_model()
        self.clients = clients
        self.total_size = 0
        for client in self.clients:
            self.total_size += client.sample_size
        self.test_loader = dataloader

    def aggregate(self):
        fed_log("Cloud server begins to aggregate client model...")
        aggregated_client_model = {}
        for client in self.clients:
            weight = client.sample_size / self.total_size
            print(f'{client.client_id}\'s weight is {weight}')
            for name, param in client.model.named_parameters():
                if name not in aggregated_client_model.keys():
                    aggregated_client_model[name] = param.data * weight
                else:
                    aggregated_client_model[name] += param.data * weight
        for name, param in self.model.named_parameters():
            param.data = aggregated_client_model[name]

        self._save_params()

    def validation(self):
        with torch.no_grad():
            test_acc, test_l = evaluate_accuracy(self.model, self.test_loader)
        return test_acc, test_l

    def _save_model(self):
        fed_log('initialize the model...')
        torch.save(self.model, FedAVG_model_path)

    def _save_params(self):
        torch.save(self.model.state_dict(), FedAVG_aggregated_model_path)


device = hp['device']
loss = nn.CrossEntropyLoss()
