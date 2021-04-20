import torch
import torch.optim as optim
from configs import HYPER_PARAMETERS as hp
from configs import FedPCA_model_path, FedPCA_aggregated_model_path
import os
import numpy as np


class Client:
    def __init__(self, client_id, dataloader, data_size):
        self.client_id = f'client_{client_id}'

        # local dataset
        self.train_loader = dataloader
        self.data_size = data_size
        self.batches = len(self.train_loader)
        # print(self.client_id, self.data_size)

        # likelihood distribution
        self.ld = np.zeros(10)
        for _, y in self.train_loader:
            self.ld[y] += 1
        self.ld = self.ld / self.ld.sum()

        # local model, optimizer
        self.model = None
        self.optimizer = None
        self.loss = torch.nn.CrossEntropyLoss()
        self.local_epochs = hp['local_epochs']

        # local update
        self.delta_w = None

        # self.current_round = 0

    def load_original_model(self):
        self.model = torch.load(FedPCA_model_path)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def initialize(self):
        if os.path.exists(FedPCA_aggregated_model_path):
            self.model.load_state_dict(torch.load(FedPCA_aggregated_model_path))

    def client_train(self):
        # print(f"{self.client_id} trains the local model...")
        current_w = torch.nn.utils.parameters_to_vector(self.model.parameters())
        for _ in range(self.local_epochs):
            for X, y in self.train_loader:
                X = X.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                train_l = self.loss(self.model(X), y)
                train_l.backward()
                self.optimizer.step()
        updated_w = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.delta_w = current_w - updated_w
        # self.current_round += 1


device = hp['device']
lr = hp['lr']
momentum = hp['momentum']
