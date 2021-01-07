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


import torch

A = torch.tensor(torch.randint(1, 9, (3, 3)), dtype=torch.float32, requires_grad=True)
B = torch.tensor(torch.randint(1, 15, (3, 3)), dtype=torch.float32, requires_grad=True)
print(f'A: {A}')
print(f'B: {B}')
C = torch.matmul(A, B)
C.sum().backward()
print(f'A.grad: {A.grad}')
print(f'B.grad: {B.grad}')
