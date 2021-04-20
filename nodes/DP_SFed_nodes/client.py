import torch
import torch.optim as optim
from _utils.calculation_utils import np_fourD_SVD, DPSGD
from configs import HYPER_PARAMETERS as hp
from configs import initial_client_model_path, client_model_path, client_outputs_path, output_grads_path
import os
from torch.utils.checkpoint import checkpoint


class Client:
    def __init__(self, client_id, dataloader, data_size):
        self.client_id = f'client_{client_id}'
        # local dataset
        self.train_loader = dataloader
        self.data_size = data_size
        self.batches = len(self.train_loader)
        # print(self.client_id, self.data_size)

        # likelihood distribution
        self.likelihood_distribution = torch.zeros(10)
        for _, y in self.train_loader:
            self.likelihood_distribution[y] += 1
        self.likelihood_distribution = self.likelihood_distribution / self.likelihood_distribution.sum()

        # participating round for each client
        self.current_round = 0
        self.participate_example_size = int(self.data_size * sampling_pr)

        # The upper edge of each client
        self.edge_server = None

        # local model, optimizer
        self.model = None
        self.optimizer = None

        # Forward pass of local model
        self.output = None

    def set_edge_server(self, edge_server):
        self.edge_server = edge_server

    def load_original_model(self):
        self.model = torch.load(initial_client_model_path)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer = DPSGD(l2_norm_clip=l2_norm_clip, noise_multiplier=sigma,
                               batch_size=self.participate_example_size, device=device,
                               params=self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def initialize(self):
        if os.path.exists(client_model_path):
            self.model.load_state_dict(torch.load(client_model_path))

    def client_forward(self):
        # print(f"{self.client_id} conducts forward propagation...")
        for batch, (X, y) in enumerate(self.train_loader):
            if batch % self.batches == self.current_round % self.batches:
                X = X.to(device)
                X.requires_grad_(True)
                self.output = checkpoint(self.model, X)
                if compress_ratio < 0.5:
                    # Compress
                    U, S, VT, Transitions = np_fourD_SVD(self.output.cpu().detach().numpy(), compress_ratio=compress_ratio)
                    self.send_to_edge((U, S, VT, y))
                    # Compress End
                    Transitions = torch.tensor(Transitions, dtype=torch.float32).to(device)
                    self.output = torch.matmul(Transitions, self.output)
                else:
                    self.send_to_edge((self.output, y))
            else:
                continue
        self.current_round += 1

    def client_backward(self):
        # print(f'{self.client_id} conducts backward propagation')
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
weight_decay = hp['weight_decay']
compress_ratio = hp['compress_ratio']
sampling_pr = hp['sampling_pr']
l2_norm_clip = hp['l2_norm_clip']
sigma = hp['sigma']
