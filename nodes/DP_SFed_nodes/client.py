"""
client in DP_SFed
"""
import torch
import torch.optim as optim
from utils import FED_LOG as fed_log
from configs import HYPER_PARAMETERS as hp
from configs import initial_client_model_path, client_model_path, client_outputs_path, output_grads_path
import os
from torch.utils.checkpoint import checkpoint


class Client:
    def __init__(self, client_id, dataloader, sample_size):
        self.client_id = f'client_{client_id}'
        self.train_loader = dataloader
        self.data_from_edge = None
        self.current_round = 0
        self.batches = len(self.train_loader)
        self.sample_size = sample_size
        self.edge_server = None
        self.model = None
        self.optimizer = None
        self.output = None

        print(self.client_id, self.sample_size)

    def set_edge_server(self, edge_server):
        self.edge_server = edge_server

    def load_original_model(self):
        self.model = torch.load(initial_client_model_path)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def initialize(self):
        if os.path.exists(client_model_path):
            self.model.load_state_dict(torch.load(client_model_path))

    def client_forward(self):
        # fed_log(f"{self.client_id} conducts forward propagation...")
        for batch, (X, y) in enumerate(self.train_loader):
            if batch % self.batches == self.current_round % self.batches:
                X = X.to(device)
                X.requires_grad_(True)
                self.output = checkpoint(self.model, X)
                self.send_to_edge((self.output, y))
            else:
                continue
        self.current_round += 1

    def client_backward(self):
        # fed_log(f'{self.client_id} conducts backward propagation')
        output_grad = torch.load(os.path.join(output_grads_path, f'{self.edge_server.edge_id}_to_{self.client_id}.pt'))
        self.optimizer.zero_grad()
        self.output.backward(output_grad)
        self.optimizer.step()

    def send_to_edge(self, out):
        torch.save(out,
                   os.path.join(client_outputs_path, f'{self.client_id}_to_{self.edge_server.edge_id}.pt'))


device = hp['device']
lr = hp['lr']
momentum = hp['momentum']
