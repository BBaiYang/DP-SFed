"""
Cloud server
"""
import torch
import torch.nn as nn
from models.model_factory import model_factory
from configs import HYPER_PARAMETERS as hp
from configs import DATA_SET_NAME as data_set_name
from configs import MODEL_NAME as model_name
from configs import initial_client_model_path, client_model_path, initial_edge_model_path, edge_model_path
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


def evaluate_accuracy(client_model, edge_model, data_iter):
    client_model.eval()
    edge_model.eval()
    metric = Accumulator(3)
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        y_pred = edge_model(client_model(X))
        test_acc = accuracy(y_pred, y)
        test_l = loss(y_pred, y)
        metric.add(test_acc, test_l * len(y), len(y))
    client_model.train()
    edge_model.train()
    return metric[0] / metric[2], metric[1] / metric[2]


class Cloud:
    def __init__(self, edges, dataloader):
        self.model = model_factory(data_set_name, model_name).to(device)
        self.client_model, self.edge_model = self._split_model()
        print(f'model:\n{self.model}\nclient_model:\n{self.client_model}\nedge_model:\n{self.edge_model}')
        self._save_model()
        self.clients = None
        self.edges = edges
        self.total_client_data_size = 0
        self.total_edge_data_size = 0
        self.test_loader = dataloader

    def initialize(self):
        for k, edge in enumerate(self.edges):
            self.total_edge_data_size += edge.sample_size
            if k == 0:
                self.clients = edge.participating_clients
            else:
                self.clients += edge.participating_clients
        for client in self.clients:
            self.total_client_data_size += client.sample_size

    def aggregate(self):
        # fed_log("Cloud server begins to aggregate client model...")
        aggregated_client_model = {}
        for k, client in enumerate(self.clients):
            weight = client.sample_size / self.total_client_data_size
            for name, param in client.model.state_dict().items():
                if k == 0:
                    aggregated_client_model[name] = param.data * weight
                else:
                    aggregated_client_model[name] += param.data * weight
        self.client_model.load_state_dict(aggregated_client_model)

        # fed_log("Cloud server begins to aggregate edge model...")
        aggregated_edge_model = {}
        local_running_means = {}
        sample_sizes = {}
        for k, edge in enumerate(self.edges):
            sample_sizes[edge.edge_id] = edge.sample_size
            weight = edge.sample_size / self.total_edge_data_size
            for name, param in edge.aggregated_model.state_dict().items():
                if k == 0:
                    aggregated_edge_model[name] = param.data * weight
                    if 'running_mean' in name:
                        local_running_means[name] = [param.data]
                else:
                    aggregated_edge_model[name] += param.data * weight
                    if 'running_mean' in name:
                        local_running_means[name] += [param.data]
        self.edge_model.load_state_dict(aggregated_edge_model)

        self._save_params()

        self.total_edge_data_size = 0
        self.total_client_data_size = 0

    def validation(self):
        with torch.no_grad():
            test_acc, test_l = evaluate_accuracy(self.client_model, self.edge_model, self.test_loader)
        return test_acc, test_l

    def _split_model(self):
        client_model = None
        edge_model = None
        if 'CNN' in model_name:
            client_model = nn.Sequential(*list(self.model.children())[:1])
            edge_model = nn.Sequential(*list(self.model.children())[1:])
        if 'VGG' in model_name:
            client_model = list(self.model.children())[0][:1]
            edge_model = list(self.model.children())[0][1:]
        return client_model, edge_model

    def _save_model(self):
        fed_log('initialize the model...')
        torch.save(self.client_model, initial_client_model_path)
        torch.save(self.edge_model, initial_edge_model_path)

    def _save_params(self):
        torch.save(self.client_model.state_dict(), client_model_path)
        torch.save(self.edge_model.state_dict(), edge_model_path)

    # def _rectify_running_var(self, local_running_means, running_mean, sample_sizes):


device = hp['device']
loss = nn.CrossEntropyLoss()
lr = hp['lr']
momentum = hp['momentum_for_BN']
