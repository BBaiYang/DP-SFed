HYPER_PARAMETERS = {

    'device': 'cuda:4',
    'device_for_baseline': 'cuda:8',

    'communication_rounds': 3000,
    'record_step': 10,
    'global_epochs': 100,
    'n_clients': 200,
    'n_edges': 3,
    'classes_per_client': 6,
    'balancedness': 1,

    'lr': .016,
    'momentum': .9,
    'bn_momentum': .1,
    'weight_decay': 5e-5,

    # FashionMNIST, CIFAR10
    'dataset': 'CIFAR10',

    # DP_SFed, FedAVG, FedPCA
    'training_strategy': 'FedPCA',

    # DP_SFed,
    'sampling_pr': 0.03,
    'sigma': 2,
    'participating_ratio': .2,
    'compress_ratio': 1,
    # fixed: CIFAR10: .1, 10 FMNIST: unknown
    'l2_norm_clip': .1,
    'edge_epochs': 5,

    # FedAVG
    'batch_size': 100,
    'local_epochs': 3,
    # FashionMNIST : CNN
    # CIFAR10 : VGG16, ResNet18, VGG11, VGG11s
    'model_name': 'VGG11s',

    # Path of ijcai_results
    'results_path': 'new_results',

    # number of clusters
    'n_cluster': 6,

    'sparse_ratio': 0.3
}
# ======================================

'''Path of various models'''
initial_client_model_path = 'intermediate_variables/initial_client_model.pt'
initial_edge_model_path = 'intermediate_variables/initial_edge_model.pt'
client_model_path = 'intermediate_variables/client_model.pt'
edge_model_path = 'intermediate_variables/edge_model.pt'
client_outputs_path = 'intermediate_variables/client_outputs'
output_grads_path = 'intermediate_variables/output_grads'
edge_aggregated_model_path = 'intermediate_variables/edge_aggregated_model.pt'
FedAVG_model_path = 'intermediate_variables/FedAVG_model.pt'
FedAVG_aggregated_model_path = 'intermediate_variables/FedAVG_aggregated_model.pt'
FedPCA_model_path = 'intermediate_variables/FedPCA_model.pt'
FedPCA_aggregated_model_path = 'intermediate_variables/FedPCA_aggregated_model.pt'
