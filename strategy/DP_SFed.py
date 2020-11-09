"""
联邦学习训练器，整体的学习策略在此类实现，如客户端选择策略、服务器节点之间服务器-客户端的通信策略等
具体的客户端训练方案、通信内容等微观层面的行为在Client，Server类中定义
"""
from DP_SFed_nodes.client import Client
from DP_SFed_nodes.edge import Edge
from DP_SFed_nodes.cloud import Cloud
from configs import MODEL_NAME as model_name
from configs import HYPER_PARAMETERS as hp
from configs import client_outputs_path, output_grads_path
from utils import FED_LOG as fed_log
from data_utils import get_data_loaders
import os


class DPSFedTrainer:
    def __init__(self):
        fed_log("A DPSFedTrainer is inited, this trainer use the strategy DP_SFed proposed by YangHe. \n")
        # 构造客户端及相应的本地数据集
        local_dataloaders, local_sample_sizes, test_dataloader = get_data_loaders()
        self.clients = [Client(_id + 1, local_dataloaders[_id], sample_size)
                        for sample_size, _id in zip(local_sample_sizes, list(range(clients_num)))]

        clients_per_edge = clients_num // edges_num
        clients_for_edges = []
        for i in range(0, clients_num, clients_per_edge):
            if len(clients_for_edges) < edges_num - 1:
                clients_for_edges.append(self.clients[i: min(i + clients_per_edge, clients_num)])
            else:
                clients_for_edges.append(self.clients[i:])
                break
        self.edges = [Edge(_id + 1, clients_for_edges[_id])
                      for _id in range(edges_num)]

        self.cloud = Cloud(self.edges, test_dataloader)
        self.communication_rounds = hp['communication_rounds']
        self.current_round = 0
        self.results = {'loss': [], 'accuracy': []}

        for client in self.clients:
            client.load_original_model()

        for edge in self.edges:
            edge.load_original_model()

    def begin_train(self):
        fed_log(f"DPSFedTrainer is going to train, the model is a: {model_name} model")
        for t in range(self.communication_rounds):
            self.current_round = t + 1
            self.train_step()
            self.validation_step()
            clear_dir(client_outputs_path)
            clear_dir(output_grads_path)

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
        fed_log(
            f"[Round: {self.current_round: 04}] Test set: Average loss: {test_l:.4f}, Accuracy: {test_acc:.4f}"
        )
        self.results['loss'].append(test_l)
        self.results['accuracy'].append(test_acc)


def clear_dir(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))


clients_num = hp['n_clients']
edges_num = hp['n_edges']
