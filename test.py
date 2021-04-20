import torch
import numpy as np
from kmeans_pytorch import kmeans
from scipy.linalg import eigh as largest_eigh
import matplotlib.pyplot as plt
import numpy as np

# # data
# data_size, dims, num_clusters = 1000, 2, 3
# x = np.random.randn(data_size, dims) / 6
# x = torch.from_numpy(x)
#
# # kmeans
# cluster_ids_x, cluster_centers = kmeans(
#     X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
# )
#
# print(cluster_ids_x, cluster_centers)

# x = torch.arange(1, 3)
# y = torch.arange(3, 5)
# x = x.view(-1, 1)
# y = y.view(-1, 1)
# z = torch.cat((x, y), dim=-1)
# print(z)

# a = torch.randint(1, 20, (10, 2))
# print(a)
# b = [0, 0, 1, 1, 0, 2, 1, 2, 2, 0]
# c = [[] for i in range(3)]
# for i, cluster in enumerate(b):
#     # print(cluster)
#     # print(c[cluster])
#     c[cluster].append(i)
# for idx in c:
#     print(a[idx, :])

# A = np.random.randn(1, 10)
# print(A)
# U, S, VT = np.linalg.svd(A)
# print(U, S, VT)
#
# N = A.shape[0]
# k = 1
# X = A @ A.T
# evals_large, evecs_large = largest_eigh(X, eigvals=(N - k, N - 1))
# print(evals_large, evecs_large)
# print((evecs_large.T @ A / np.sqrt(evals_large)).T)

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
# x = wasserstein_distance(u_values, u_values,
#                          [0, 10, 10, 10, 10, 0, 0, 0, 0, 0], [0, 20, 20, 20, 20, 0, 0, 0, 0, 0])
# y = jensenshannon([0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0], [0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0])
# print("x = ", x, "y = ", y)



