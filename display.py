import torch
import matplotlib.pyplot as plt

x = list(range(10, 2001, 10))
result1 = torch.load('new_results/CIFAR10_FedPCA_n_clients_200_participating_ratio_0.2_n_cluster_6.pt')['accuracy']
result2 = torch.load('new_results/CIFAR10_FedAVG_n_clients_200_participating_ratio_0.2.pt')['accuracy']
result3 = torch.load('new_results/CIFAR10_FedPCA2_n_clients_200_participating_ratio_0.2_n_cluster_6.pt')['accuracy']
# for i, (acc1, acc2) in enumerate(zip(result1, result2)):
#     print((i + 1) * 2, acc1, acc2)
plt.plot(x, result1, label='FedPCA')
plt.plot(x, result2, label='FedAVG')
plt.plot(x, result3, label='FedPCA2')
plt.legend(loc='best')
plt.savefig('FedPCA_clients_200.png')
# for i, acc in enumerate(result2):
#     print((i + 1) * 10, acc)
