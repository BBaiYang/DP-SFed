HYPER_PARAMETERS = {

    'device': 'cuda:1',
    'device_for_baseline': 'cuda:8',

    'communication_rounds': 2000,
    'record_step': 10,
    'global_epochs': 100,
    'n_clients': 100,
    'n_edges': 3,
    'classes_per_client': 3,
    'balancedness': .995,

    'lr': .016,
    'momentum': .9,
    'bn_momentum': .1,
    'weight_decay': 5e-5,

    # FashionMNIST, CIFAR10
    'dataset': 'CIFAR10',

    # DP_SFed, FedAVG
    'training_strategy': 'DP_SFed',

    # DP_SFed,
    'sampling_pr': 0.03,
    'sigma': 2,
    'participating_ratio': .5,
    'compress_ratio': 1,
    # fixed: CIFAR10: .1, 10 FMNIST: unknown
    'l2_norm_clip': .1,
    'edge_epochs': 5,

    # FedAVG
    'batch_size': 200,
    'local_epochs': 10,

    # FashionMNIST : CNN
    # CIFAR10 : VGG16, ResNet18, VGG11, VGG11s
    'model_name': 'VGG16',

    # Path of results
    'results_path': 'new_results'
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
