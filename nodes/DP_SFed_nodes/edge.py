"""
edge server
"""
from utils import FED_LOG as fed_log
import torch.optim as optim
from configs import initial_edge_model_path, edge_model_path, client_outputs_path, output_grads_path
from configs import HYPER_PARAMETERS as hp
import torch
import os
import random


class Edge:
    def __init__(self, edge_id, clients):
        self.edge_id = f'edge_{edge_id}'
        self.clients = clients
        self.sample_size = 0
        self.models = {}
        self.optimizers = {}
        self.aggregated_model = None
        self.optimizer = None
        self.participating_clients = None
        for client in self.clients:
            client.set_edge_server(self)

    def load_original_model(self):
        self.aggregated_model = torch.load(initial_edge_model_path)
        self.optimizer = optim.SGD(self.aggregated_model.parameters(), lr=lr, momentum=momentum)

    def initialize(self):
        self.sample_size = 0
        if os.path.exists(edge_model_path):
            self.aggregated_model.load_state_dict(torch.load(edge_model_path))
        self.participating_clients = random.sample(self.clients, int(participating_ratio * len(self.clients)))
        for client in self.participating_clients:
            self.sample_size += client.sample_size

    # def edge_forward_backward(self):
    #     for client in self.clients:
    #         client_id = client.client_id
    #         fed_log(f'{self.edge_id} uses data from {client_id}...')
    #         X, y = torch.load(os.path.join(client_outputs_path, f'{client_id}_to_{self.edge_id}.pt'))
    #         # X.retain_grad()
    #         self.optimizers[client_id].zero_grad()
    #         train_l = loss(self.models[client_id](X), y.to(device))
    #         train_l.backward()
    #         self.optimizers[client_id].step()
    #         # torch.save(X.grad, os.path.join(output_grads_path, f'{self.edge_id}_to_{client_id}.pt'))
    #     self.aggregate_client_models()

    # def aggregate_client_models(self):
    #     aggregated_weight = {}
    #     for k, client in enumerate(self.clients):
    #         client_id = client.client_id
    #         weight = client.sample_size / self.sample_size
    #         for name, param in self.models[client_id].named_parameters():
    #             if k == 0:
    #                 aggregated_weight[name] = param.data * weight
    #             else:
    #                 aggregated_weight[name] += param.data * weight
    #     for name, param in self.aggregated_model.named_parameters():
    #         param.data = aggregated_weight[name]

    def edge_forward_backward(self):
        features = None
        labels = None
        client_step_sizes = []
        for _ in range(edge_epochs):
            if not (features, labels) == (None, None):
                features, labels = None, None
            for client in self.participating_clients:
                client_id = client.client_id
                # fed_log(f'{self.edge_id} uses data from {client_id}...')
                X, y = torch.load(os.path.join(client_outputs_path, f'{client_id}_to_{self.edge_id}.pt'))
                client_step_sizes.append(len(X))
                if (features, labels) == (None, None):
                    features, labels = X, y.to(device)
                else:
                    features, labels = torch.cat((features, X)), torch.cat((labels, y.to(device)))
            features.retain_grad()
            self.optimizer.zero_grad()
            train_l = loss(self.aggregated_model(features), labels)
            train_l.backward()
            self.optimizer.step()

        self.send_to_client(features.grad.data, client_step_sizes)

    def send_to_client(self, output_grad, client_step_sizes):
        start = 0
        end = 0
        for i, client in enumerate(self.participating_clients):
            client_id = client.client_id
            step_size = client_step_sizes[i]
            end += step_size
            torch.save(output_grad[start: end],
                       os.path.join(output_grads_path, f'{self.edge_id}_to_{client_id}.pt'))
            start = end


device = hp['device']
lr = hp['lr']
momentum = hp['momentum']
loss = torch.nn.CrossEntropyLoss()
edge_epochs = hp['edge_epochs']
participating_ratio = hp['participating_ratio']
