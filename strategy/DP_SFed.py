"""
联邦学习训练器，整体的学习策略在此类实现，如客户端选择策略、服务器节点之间服务器-客户端的通信策略等
具体的客户端训练方案、通信内容等微观层面的行为在Client，Server类中定义
"""
from nodes.DP_SFed_nodes.client import Client
from nodes.DP_SFed_nodes.edge import Edge
from nodes.DP_SFed_nodes.cloud import Cloud
from configs import HYPER_PARAMETERS as hp
from configs import client_outputs_path, output_grads_path
from data.data_utils import get_data_loaders
import os
import torch
from _utils.calculation_utils import kmeans


class DPSFedTrainer:
    def __init__(self):
        print("A DPSFedTrainer is inited, this trainer use the strategy DP_SFed proposed by YangHe. \n")
        # construct local dataset
        local_dataloaders, local_sample_sizes, test_dataloader = get_data_loaders()
        self.clients = [Client(_id + 1, local_dataloaders[_id], sample_size)
                        for sample_size, _id in zip(local_sample_sizes, list(range(clients_num)))]

        # use K-means to cluster the clients and assign them to different edgess
        clf = kmeans(self.clients, 4)
        proportion = 1 / edges_num
        clients_for_edges = [[] for i in range(0, edges_num)]
        for indices in clf.values():
            clients_per_edge = int(len(indices) * proportion)
            for i, j in enumerate(range(0, len(indices), clients_per_edge)):
                if i < edges_num - 1:
                    idxes = indices[j: j+clients_per_edge]
                    for idx in idxes:
                        clients_for_edges[i].append(self.clients[idx])
                else:
                    idxes = indices[j:]
                    for idx in idxes:
                        clients_for_edges[i].append(self.clients[idx])
                    break
        # for cs in clients_for_edges:
        #     print([c.client_id for c in cs])

        # edges
        self.edges = [Edge(_id + 1, clients_for_edges[_id])
                      for _id in range(edges_num)]

        # cloud
        self.cloud = Cloud(self.edges, test_dataloader)
        self.communication_rounds = hp['communication_rounds']
        self.current_round = 0
        self.results = {'loss': [], 'accuracy': []}

        for client in self.clients:
            client.load_original_model()

        for edge in self.edges:
            edge.load_original_model()

    def begin_train(self):
        print(f"DPSFedTrainer is going to train, the model is a: {model_name} model")
        for t in range(self.communication_rounds):
            self.current_round = t + 1
            self.train_step()
            self.validation_step()
            clear_dir(client_outputs_path)
            clear_dir(output_grads_path)
        torch.save(self.results, os.path.join('new_results',
                                              f'{dataset_name}_'
                                              f'{training_strategy}_'
                                              f'participating_ratio_{participating_ratio}_'
                                              f'sampling_pr_{sampling_pr}_'
                                              f'compression_ratio_{compression_ratio}_'
                                              f'sigma_{sigma}.pt'))

    def train_step(self):
        """客户端选择阶段"""
        for edge in self.edges:
            edge.initialize()
        self.cloud.initialize()

        """训练阶段"""
        for client in self.cloud.clients:
            client.initialize()
            client.client_forward()
        for edge in self.edges:
            edge.edge_forward_backward()
        for client in self.cloud.clients:
            client.client_backward()
        self.cloud.aggregate()

    def validation_step(self):
        test_acc, test_l = self.cloud.validation()
        print(
            f"[Round: {self.current_round: 04}] Sample size: {self.cloud.total_client_data_size},"
            f"Test set: Average loss: {test_l:.4f}, Accuracy: {test_acc:.4f}"
        )
        if self.current_round % record_step == 0:
            self.results['loss'].append(test_l)
            self.results['accuracy'].append(test_acc)


def clear_dir(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))


clients_num = hp['n_clients']
edges_num = hp['n_edges']
record_step = hp['record_step']
training_strategy = hp['training_strategy']
sampling_pr = hp['sampling_pr']
participating_ratio = hp['participating_ratio']
edge_epochs = hp['edge_epochs']
compression_ratio = hp['compress_ratio']
dataset_name = hp['dataset']
sigma = hp['sigma']
model_name = hp['model_name']
