"""
client in DP_SFed
"""
import torch
import torch.optim as optim
from utils import FED_LOG as fed_log
from configs import HYPER_PARAMETERS as hp
from configs import FedAVG_model_path, FedAVG_aggregated_model_path
import os


class Client:
    def __init__(self, client_id, dataloader, sample_size):
        self.client_id = f'client_{client_id}'
        self.train_loader = dataloader
        self.current_round = 0
        self.batches = len(self.train_loader)
        self.sample_size = sample_size
        self.model = None
        self.optimizer = None
        self.loss = torch.nn.CrossEntropyLoss()
        self.local_epochs = hp['local_epochs']

        print(self.client_id, self.sample_size)

    def load_original_model(self):
        self.model = torch.load(FedAVG_model_path)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def initialize(self):
        if os.path.exists(FedAVG_aggregated_model_path):
            self.model.load_state_dict(torch.load(FedAVG_aggregated_model_path))

    def client_train(self):
        fed_log(f"{self.client_id} trains the local model...")
        for _ in range(self.local_epochs):
            for batch, (X, y) in enumerate(self.train_loader):
                if batch % self.batches == self.current_round % self.batches:
                    X = X.to(device)
                    y = y.to(device)
                    self.optimizer.zero_grad()
                    train_l = self.loss(self.model(X), y)
                    train_l.backward()
                    self.optimizer.step()
                else:
                    continue
        self.current_round += 1


device = hp['device']
lr = hp['lr']
momentum = hp['momentum']
