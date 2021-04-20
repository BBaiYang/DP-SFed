from strategy.FedAVG import FedAVGTrainer
from strategy.DP_SFed import DPSFedTrainer
from strategy.FedPCA import FedPCATrainer
from configs import HYPER_PARAMETERS as hp

trainer = None
training_strategy = hp['training_strategy']
if training_strategy == 'DP_SFed':
    trainer = DPSFedTrainer()
elif training_strategy == 'FedAVG':
    trainer = FedAVGTrainer()
elif training_strategy == 'FedPCA':
    trainer = FedPCATrainer()
trainer.begin_train()


# from scipy.stats import wasserstein_distance
# from scipy.spatial.distance import jensenshannon
# import numpy as np
# u_values = np.arange(0, 10)
# print(u_values)
# x = wasserstein_distance([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                          [0, 0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0], [0, 0.3, 0.3, 0, 0.4, 0, 0, 0, 0, 0])
# y = jensenshannon([0, 0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0], [0, 0.3, 0.3, 0, 0.4, 0, 0, 0, 0, 0])
# print("x = ", x, "y = ", y)

# import torch
# ratio = 0.2
# a = torch.randn(10)
# print(a)
# print(torch.abs(a))
# zero = torch.zeros_like(a)
# idx = round(len(a) * ratio)
# _, indices = torch.sort(torch.abs(a))
# threshold = torch.abs(a[indices[idx]])
# a = torch.where(torch.abs(a) < threshold, zero, a)
# print(a)



