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
    'device': 'cuda: 9',
    'device_for_baseline': 'cuda: 4',
    # FashionMNIST, CIFAR10
    'dataset': 'CIFAR10',

    'communication_rounds': 2000,
    'record_step': 10,
    'global_epochs': 100,
    'n_clients': 100,
    'n_edges': 3,
    'classes_per_client': 3,
    'balancedness': .995,

    'lr': .015,
    'momentum': .9,
    'bn_momentum': .1,

    # DP_SFed, FedAVG
    'training_strategy': 'DP_SFed',

    # DP_SFed,
    'sampling_pr': .03,
    'sigma': 20,
    'participating_ratio': .5,
    'compress_ratio': .3,
    # fixed: CIFAR10: .1, 10 FMNIST: unknown
    'l2_norm_clip': .1,
    'edge_epochs': 10,

    # FedAVG
    'batch_size': 200,
    'local_epochs': 10,
}
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
#            FashionMNIST : CNN
#            CIFAR10 : VGG16, ResNet18

MODEL_NAME = 'VGG16'
# ======================================
