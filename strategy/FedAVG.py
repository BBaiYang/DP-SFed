"""
联邦学习训练器，整体的学习策略在此类实现，如客户端选择策略、服务器节点之间服务器-客户端的通信策略等
具体的客户端训练方案、通信内容等微观层面的行为在Client，Server类中定义
"""
from nodes.FedAVG_nodes.client import Client
from nodes.FedAVG_nodes.cloud import Cloud
from configs import HYPER_PARAMETERS as hp
from data.data_utils import get_data_loaders
import torch
import os


class FedAVGTrainer:
    def __init__(self):
        print("A FedAVGTrainer is inited. \n")
        clients_num = hp['n_clients']

        # 构造客户端及相应的本地数据集
        local_dataloaders, local_sample_sizes, test_dataloader = get_data_loaders()
        self.clients = [Client(_id + 1, local_dataloaders[_id], sample_size)
                        for sample_size, _id in zip(local_sample_sizes, list(range(clients_num)))]

        self.cloud = Cloud(self.clients, test_dataloader)
        self.communication_rounds = hp['communication_rounds']
        self.current_round = 0
        self.results = {'loss': [], 'accuracy': []}

        for client in self.clients:
            client.load_original_model()

    def begin_train(self):
        print(f"FedAVGTrainer is going to train, the model is a: {model_name} model")
        for t in range(self.communication_rounds):
            self.current_round = t + 1
            self.train_step()
            self.validation_step()
        torch.save(self.results, os.path.join(results_path,
                                              f'{dataset_name}_'
                                              f'{training_strategy}_'
                                              f'participating_ratio_{participating_ratio}.pt'))

    def train_step(self):
        """客户端选择阶段"""
        self.cloud.initialize()

        """训练阶段"""
        for client in self.cloud.participating_clients:
            client.initialize()
            client.client_train()
        self.cloud.aggregate()

    def validation_step(self):
        test_acc, test_l = self.cloud.validation()
        print(
            f"[Round: {self.current_round: 04}] Test set: Average loss: {test_l:.4f}, Accuracy: {test_acc:.4f}"
        )
        if self.current_round % record_step == 0:
            self.results['loss'].append(test_l)
            self.results['accuracy'].append(test_acc)


participating_ratio = hp['participating_ratio']
training_strategy = hp['training_strategy']
record_step = hp['record_step']
dataset_name = hp['dataset']
results_path = hp['results_path']
model_name = hp['model_name']
