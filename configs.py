"""
配置文件，实验前需要在此设定一些参数
"""

"""日志打印能力设置"""
# ======================================
# DEBUG值为1时在程序运行过程中会打印出日志信息，为0则关闭打印
DEBUG = 1
# ======================================

"""超参数设置,字典方式存储"""
# ======================================
HYPER_PARAMETERS = {
    'device': 'cuda: 3',
    'device_for_baseline': 'cuda: 2',
    'communication_rounds': 2000,
    'global_epochs': 100,
    'edge_epochs': 20,
    'lr': .01,
    'momentum_for_trainer': .9,
    'momentum_for_BN': .1,
    'sampling_pr': .02,
    'batch_size': 250,
    'n_clients': 10,
    'n_edges': 3,
    'classes_per_client': 4,
    'balancedness': .7,
    'dataset': 'CIFAR10',
    'local_epochs': 1,
    'participating_ratio': 1
}
# ======================================

"""原始数据集设置"""
# ======================================
# 可用参数：MINIST FashionMNIST CIFAR10
DATA_SET_NAME = 'CIFAR10'
# ======================================

'''模型保存路径'''
initial_client_model_path = 'intermediate_variables/initial_client_model.pt'
initial_edge_model_path = 'intermediate_variables/initial_edge_model.pt'
client_model_path = 'intermediate_variables/client_model.pt'
edge_model_path = 'intermediate_variables/edge_model.pt'
client_outputs_path = 'intermediate_variables/client_outputs'
output_grads_path = 'intermediate_variables/output_grads'
edge_aggregated_model_path = 'intermediate_variables/edge_aggregated_model.pt'
FedAVG_model_path = 'intermediate_variables/FedAVG_model.pt'
FedAVG_aggregated_model_path = 'intermediate_variables/FedAVG_aggregated_model.pt'

"""模型结构设置"""
# ======================================
# 首先需要在这里先设计好最终所需要模型的整体结构，将此类实例化作为参数传给联邦学习训练器
# 因为描述模型设计参数较多，不适合设计成传参再构造网络的方式，这里就直接编辑类定义再实例化
# 可用参数：
#            MINIST : CNN
#            FashionMNIST : CNN
#            CIFAR10 : VGG16

MODEL_NAME = 'VGG16'
# ======================================
