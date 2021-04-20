import torch
import torch.nn as nn
from models.model_factory import model_factory
from configs import HYPER_PARAMETERS as hp
from configs import FedPCA_model_path, FedPCA_aggregated_model_path
import random
from sklearn.cluster import AgglomerativeClustering
from scipy.linalg import eigh as largest_eigh
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
import numpy as np


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


def sim(x, y):
    return wasserstein_distance(np.arange(0, 10), np.arange(0, 10), x, y)


def sim_affinity(X):
    return pairwise_distances(X, metric=sim)


class Cloud:
    def __init__(self, clients, dataloader):
        # global model
        self.model = model_factory(data_set_name, model_name).to(device)
        print(self.model)
        self._save_model()

        # clients
        self.clients = clients
        self.total_size = 0

        # test dataset
        self.test_loader = dataloader

        # participating clients
        self.participating_clients = None

        # communication round
        self.current_round = 1

        # cluster all the clients according to Earth Mover's Distance
        self.clusters = [[] for _ in range(n_cluster)]
        lds = None
        for k, client in enumerate(self.clients):
            if k == 0:
                lds = client.ld
            else:
                lds = np.vstack((lds, client.ld))
        ac = AgglomerativeClustering(n_clusters=n_cluster, affinity=sim_affinity, linkage='average')
        ld_clustering = ac.fit(lds)
        ld_result = ld_clustering.labels_
        for i, cluster in enumerate(ld_result):
            self.clusters[cluster].append(i)
        for cluster, clients in enumerate(self.clusters):
            print(f'{cluster}({len(clients)}): {clients}')

    def initialize(self):

        # Participants Selection
        self.total_size = 0
        self.participating_clients = []
        for clients in self.clusters:
            num = round(participating_ratio * len(clients))
            self.participating_clients.extend(random.sample(clients, num))
        for client_id in self.participating_clients:
            self.total_size += self.clients[client_id].data_size
        # print(f'********************* '
        #       f'Participants in round {self.current_round}: {self.participating_clients}, '
        #       f'Total size in round {self.current_round}: {self.total_size} '
        #       f'*********************')
        self.current_round += 1

        # Cluster the participants according to Earth Mover's Distance
        # participants_clusters = [[] for _ in range(n_cluster)]
        # participants_lds = None
        # for k, client_id in enumerate(self.participating_clients):
        #     if k == 0:
        #         participants_lds = self.clients[client_id].ld
        #     else:
        #         participants_lds = np.vstack((participants_lds, self.clients[client_id].ld))
        # ac = AgglomerativeClustering(n_clusters=n_cluster, affinity=sim_affinity, linkage='average')
        # ld_clustering = ac.fit(participants_lds)
        # ld_result = ld_clustering.labels_
        # for i, cluster in enumerate(ld_result):
        #     participants_clusters[cluster].append(i)
        # for cluster, clients in enumerate(participants_clusters):
        #     print(f'{cluster}({len(clients)}): {[self.participating_clients[client_id] for client_id in clients]}')

    def aggregate(self):
        data_sizes, main_directions = self._cluster_updates()
        updated_delta_w = None
        for i, (data_size, main_direction) in enumerate(zip(data_sizes, main_directions)):
            weight = data_size / self.total_size
            zero = torch.zeros_like(main_direction)
            idx = round(len(main_direction) * sparse_ratio)
            _, indices = torch.sort(torch.abs(main_direction))
            threshold = torch.abs(main_direction[indices[idx]])
            main_direction = torch.where(torch.abs(main_direction) < threshold, zero, main_direction)
            if i == 0:
                updated_delta_w = main_direction * weight
            else:
                updated_delta_w += main_direction * weight
        current_w = torch.nn.utils.parameters_to_vector(self.model.parameters())
        torch.nn.utils.vector_to_parameters(current_w - updated_delta_w.flatten(), self.model.parameters())
        self._save_params()

    def validate(self):
        with torch.no_grad():
            test_acc, test_l = evaluate_accuracy(self.model, self.test_loader)
        return test_acc, test_l

    def _save_model(self):
        print('initialize the model...')
        torch.save(self.model, FedPCA_model_path)

    def _save_params(self):
        torch.save(self.model.state_dict(), FedPCA_aggregated_model_path)

    def _cluster_updates(self):
        # Construct Matrix delta_ws, which contains updates of each participants
        delta_ws = None
        for k, client_id in enumerate(self.participating_clients):
            if k == 0:
                delta_ws = self.clients[client_id].delta_w.view(1, -1)
            else:
                delta_ws = torch.cat((delta_ws, self.clients[client_id].delta_w.view(1, -1)), dim=0)

        # Cluster updates
        ac = AgglomerativeClustering(n_clusters=n_cluster, affinity='cosine', linkage='complete')
        clustering = ac.fit(delta_ws.detach().cpu().numpy())
        cluster_result = clustering.labels_
        cluster_idx = [[] for _ in range(n_cluster)]
        for i, cluster in enumerate(cluster_result):
            cluster_idx[cluster].append(i)
        # for i, idx in enumerate(cluster_idx):
        #     c_ids = [self.participating_clients[client_id] for client_id in idx]
        #     print(f'{i}({len(c_ids)}): {c_ids}')

        # Find the main direction of each cluster
        main_directions = list()
        data_sizes = list()
        for idx in cluster_idx:
            data_size = 0
            for i in idx:
                data_size += self.clients[self.participating_clients[i]].data_size
            data_sizes.append(data_size)
            tmp = delta_ws[idx, :]
            X = torch.matmul(tmp, tmp.T)
            N = X.size(0)

            cos = torch.nn.CosineSimilarity(dim=0)
            for i in range(N - 1):
                for j in range(i + 1, N):
                    print(f'Cosine Similarity between {i} and {j}: '
                          f'{cos(tmp[i].view(-1, 1), tmp[j].view(-1, 1)).item()}')

            # if N < 3:
            #     k = 1
            # else:
            #     k = 2
            k = 1
            evals_large, evecs_large = largest_eigh(X.detach().cpu().numpy(), eigvals=(N - N, N - 1))
            evals_large_main = torch.tensor(evals_large).to(device)[N-k:]
            evecs_large_main = torch.tensor(evecs_large).to(device)[:, N-k:]

            main_component = None
            cos = torch.nn.CosineSimilarity(dim=0)
            for i in range(k):
                z = torch.matmul(evecs_large_main[:, i].view(1, -1), tmp).T / torch.sqrt(evals_large_main[i])
                positive = 0
                for _ in range(N):
                    if cos(z, tmp[i].view(-1, 1)).item() > 0:
                        positive += 1
                if positive < N - positive:
                    z = -z
                if i == 0:
                    main_component = z
                else:
                    main_component = torch.cat((main_component, z), dim=1)
            main_component = torch.matmul(main_component, evals_large_main.view(-1, 1) / evals_large_main.sum())
            main_component = main_component / torch.norm(main_component)

            # scale of main direction of each cluster
            variance = 0
            for k, i in enumerate(idx):
                variance += self.clients[self.participating_clients[i]].data_size / data_size * torch.norm(tmp[k])
            # for i in range(len(tmp)):
            #     print(f'Cosine Similarity: {cos(main_component, tmp[i].view(-1, 1)).item()}, '
            #           f'Variance: {variance}, '
            #           f'Ratio: {evals_large_main.sum() / sum(evals_large)} '
            #           f'Norm of Main Component: {torch.norm(main_component)}, '
            #           f'Norm of Origin Delta_W: {torch.norm(tmp[i])}')
            main_directions.append(main_component * variance)
        return data_sizes, main_directions

    def _pairwise_cosine(self):
        participants_num = len(self.participating_clients)
        cos = torch.nn.CosineSimilarity(dim=0)
        for i in range(participants_num - 1):
            for j in range(i + 1, participants_num):
                pairwise_cosine = cos(self.participating_clients[i].delta_w, self.participating_clients[j].delta_w)
                print(f'{self.participating_clients[i].client_id}, {self.participating_clients[j].client_id}: '
                      f'{pairwise_cosine}')


device = hp['device']
loss = nn.CrossEntropyLoss()
participating_ratio = hp['participating_ratio']
data_set_name = hp['dataset']
model_name = hp['model_name']
n_cluster = hp['n_cluster']
sparse_ratio = hp['sparse_ratio']
