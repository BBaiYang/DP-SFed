import numpy as np
#
# np.random.seed(3)
# X = np.random.rand(10, 5120)
# w = np.random.rand(10, 1)
# w = (np.exp(w)) / np.exp(w).sum()
#
# global_dir = X.T @ w
# U, S, VT = np.linalg.svd(X)
# main_direction = VT[0]
#
# new_global_dir = (X @ main_direction).T @ w * main_direction
# print('global direction', global_dir, np.linalg.norm(global_dir))
# print('new global direction', new_global_dir, np.linalg.norm(new_global_dir))
#
# print(global_dir.T @ new_global_dir / (np.linalg.norm(global_dir) * np.linalg.norm(new_global_dir)))

# import time
# from scipy.linalg import eigh as largest_eigh
# # from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
#
# np.set_printoptions(suppress=True)
# np.random.seed(0)
# N = 5000
# M = 10
# k = 1
# X = np.random.random((N, 10)) - 0.5
# A = np.dot(X.T, X)
#
# # Benchmark the dense routine
# start = time.time()
# evals_large, evecs_large = largest_eigh(A, eigvals=(M - k, M - 1))
# # print(evals_large, evecs_large)
# elapsed = (time.time() - start)
# # print("eigh elapsed time: ", elapsed)
#
# # # Benchmark the sparse routine
# # start = time.time()
# # evals_large_sparse, evecs_large_sparse = largest_eigsh(A, k, which='LM')
# # print(evals_large_sparse, evecs_large_sparse)
# # elapsed = (time.time() - start)
# # print("eigsh elapsed time: ", elapsed)
# #
# # print((evecs_large.T * evecs_large_sparse.T).sum() / np.linalg.norm(evecs_large) / np.linalg.norm(evecs_large_sparse))
#
# main_dir = X @ evecs_large / np.sqrt(evals_large)
# print('main_dir', main_dir)
#
# _, _, VT = np.linalg.svd(X.T)
# actual_dir = VT[0].reshape(-1, 1)
# print('actual_dir', actual_dir)
#
# print('cos value', main_dir.T @ actual_dir / np.linalg.norm(main_dir) / np.linalg.norm(actual_dir))
#
# print((X.T @ main_dir).mean() * main_dir)


import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils import data
from models.forCIFAR10.resnet18 import ResNet18


resnet18 = ResNet18()
# print(f'resnet18: {resnet18}')
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in resnet18.parameters())))

shallow_model = nn.Sequential(*list(resnet18.children())[:-1])
deep_model = nn.Sequential(*list(resnet18.children())[-1:])
#
# print(shallow_model)
# print(deep_model)
#
#
# if __name__ == '__main__':
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     batch_size = 10
#
#     cifar_train = torchvision.datasets.CIFAR10(root='~/CODES/Dataset',
#                                                     train=True, download=True, transform=train_transform)
#     cifar_test = torchvision.datasets.CIFAR10(root='~/CODES/Dataset',
#                                                    train=False, download=True, transform=test_transform)
#
#     train_loader = data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_loader = data.DataLoader(cifar_test, batch_size=batch_size, num_workers=4)
#
#     for X, y in train_loader:
#         print(deep_model(shallow_model(X)))
#         print(resnet18(X))
#         break
