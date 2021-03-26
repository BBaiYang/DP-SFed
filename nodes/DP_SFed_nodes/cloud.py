"""
Cloud server
"""
import torch
import torch.nn as nn
from models.model_factory import model_factory
from configs import HYPER_PARAMETERS as hp
from configs import initial_client_model_path, client_model_path, initial_edge_model_path, edge_model_path


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
        # init global model, edge-side model and client-side model
        self.model = model_factory(data_set_name, model_name).to(device)
        self.client_model, self.edge_model = self._split_model()
        print(f'model:\n{self.model}\nclient_model:\n{self.client_model}\nedge_model:\n{self.edge_model}')
        self._save_model()

        # Initial phase
        self.clients = None
        self.edges = edges
        self.total_client_data_size = 0
        self.total_edge_data_size = 0
        self.test_loader = dataloader

    def initialize(self):
        self.total_edge_data_size = 0
        self.total_client_data_size = 0
        for k, edge in enumerate(self.edges):
            self.total_edge_data_size += edge.data_size
            if k == 0:
                self.clients = edge.participating_clients[:]
            else:
                self.clients += edge.participating_clients[:]
        for client in self.clients:
            self.total_client_data_size += client.data_size

    def aggregate(self):
        # print("Cloud server begins to aggregate client model...")
        aggregated_client_model = {}
        for k, client in enumerate(self.clients):
            weight = client.data_size / self.total_client_data_size
            # print(client.client_id, client.sample_size, self.total_client_data_size, weight)
            for name, param in client.model.state_dict().items():
                if k == 0:
                    aggregated_client_model[name] = param.data * weight
                else:
                    aggregated_client_model[name] += param.data * weight
        self.client_model.load_state_dict(aggregated_client_model)

        # print("Cloud server begins to aggregate edge model...")
        aggregated_edge_model = {}
        for k, edge in enumerate(self.edges):
            weight = edge.data_size / self.total_edge_data_size
            # print(edge.edge_id, edge.sample_size, self.total_edge_data_size, weight)
            for name, param in edge.aggregated_model.state_dict().items():
                if k == 0:
                    aggregated_edge_model[name] = param.data * weight
                else:
                    aggregated_edge_model[name] += param.data * weight
        self.edge_model.load_state_dict(aggregated_edge_model)
        self._save_params()

    def validation(self):
        with torch.no_grad():
            test_acc, test_l = evaluate_accuracy(self.client_model, self.edge_model, self.test_loader)
        return test_acc, test_l

    def _split_model(self):
        if 'VGG' in model_name:
            client_model = list(self.model.children())[0][:1]
            edge_model = list(self.model.children())[0][1:]
        else:
            client_model = nn.Sequential(*list(self.model.children())[:1])
            edge_model = nn.Sequential(*list(self.model.children())[1:])
        return client_model, edge_model

    def _save_model(self):
        print('initialize the model...')
        torch.save(self.client_model, initial_client_model_path)
        torch.save(self.edge_model, initial_edge_model_path)

    def _save_params(self):
        torch.save(self.client_model.state_dict(), client_model_path)
        torch.save(self.edge_model.state_dict(), edge_model_path)


device = hp['device']
loss = nn.CrossEntropyLoss()
lr = hp['lr']
data_set_name = hp['dataset']
bn_momentum = hp['bn_momentum']
model_name = hp['model_name']
