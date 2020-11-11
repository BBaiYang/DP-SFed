import torch

result = torch.load('results/DP_SFed_sampling_pr_0.02_participating_ratio_0.7_edge_epochs_20.pt')
print(result['accuracy'])
