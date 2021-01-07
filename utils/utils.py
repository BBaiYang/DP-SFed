# -*- coding: utf-8 -*-
"""
提供一些工具函数
"""

import numpy as np
from configs import DEBUG as debug
from scipy.linalg import eigh as largest_eigh
import torch
from torch.distributions.normal import Normal
from torch.optim import SGD

"""打印详细日志"""


def FED_LOG(*kwargs):
    if debug:
        print(*kwargs)


"""输出关键信息"""


def FED_WRITE(*kwargs):
    # TODO: 向文件输入信息流
    if debug:
        print(*kwargs)


def twoD_svd(X, compress_ratio):
    # 奇异值分解
    k = int(compress_ratio * X.shape[0])
    # 数据集矩阵 奇异值分解  返回的Sigma 仅为对角线上的值
    # 已经自动排序了
    U, Sigma, VT = np.linalg.svd(X)

    # 对奇异值从大到小排序，返回索引
    indexVec = np.argsort(-Sigma)

    # 根据求得的分解，取出前k大的奇异值对应的U,Sigma,V
    K_index = indexVec[:k]  # 取出前k最大的特征值的索引

    U = U[:, K_index]  # 从U取出前k大的奇异值的对应(按列取)
    S = np.diag(Sigma[:k])
    VT = VT[K_index, :]  # 从VT取出前k大的奇异值的对应(按行取)
    return U @ S @ VT


# np_threeD_SVD(X, compress_ratio) is for 3D matrix with shape (m,n,c)
# It is compressed in the channels of (m,n)
def np_threeD_SVD(X, compress_ratio):
    k = int(compress_ratio * X.shape[0])
    (m, n, c) = X.shape

    U = np.zeros((m, k, c))
    S = np.zeros((k, c))
    VT = np.zeros((k, n, c))
    for j in range(c):
        pU, pSigma, pVT = np.linalg.svd(X[:, :, j])

        pU = pU[:, list(range(0, k))]  # 从U取出前k大的奇异值的对应(按列取)
        U[:, :, j] = pU

        S[:, j] = pSigma[:k]

        pVT = pVT[list(range(0, k)), :]  # 从VT取出前k大的奇异值的对应(按行取)
        VT[:, :, j] = pVT

    return U, S, VT


# With respect to np_threeD_SVD(X, compress_ratio)
# It restores the original matrix from compressed vectors
def np_threeD_compound(U, S, VT):
    (k, c) = S.shape
    m = U.shape[0]
    n = VT.shape[1]
    SS = np.zeros((k, k, c))
    X = np.zeros((m, n, c))
    for i in range(c):
        for j in range(k):
            SS[j][j][i] = S[j, i]  # 奇异值list形成矩阵

    for i in range(c):
        pX = np.dot(U[:, :, i], SS[:, :, i])
        pX = np.dot(pX, VT[:, :, i])
        X[:, :, i] = pX
    return X


# np_fourD_SVD(X, compress_ratio) is for 4D matrix with shape (batch_size,c, m, n)
# It is compressed in the channels of (m,n)
def np_fourD_SVD(X, compress_ratio):
    (bs, c, m, n) = X.shape
    k = int(compress_ratio * m)
    U = np.zeros((bs, c, m, k))
    S = np.zeros((bs, c, k))
    VT = np.zeros((bs, c, k, n))
    Transitions = np.zeros((bs, c, m, m))
    E = np.eye(k)
    E = np.pad(E, ((0, m - k), (0, m - k)), 'constant', constant_values=(0, 0))
    for i in range(bs):
        for j in range(c):
            pU, pSigma, pVT = np.linalg.svd(X[i, j, :, :])
            Transitions[i, j, :, :] = pU @ E @ pU.T
            pU = pU[:, list(range(0, k))]  # 从U取出前k大的奇异值的对应(按列取)
            U[i, j, :, :] = pU
            S[i, j, :] = pSigma[:k]
            pVT = pVT[list(range(0, k)), :]  # 从VT取出前k大的奇异值的对应(按行取)
            VT[i, j, :, :] = pVT
    return U, S, VT, Transitions


# With respect to np_fourD_SVD(X, compress_ratio)
# It restores the original matrix from compressed vectors
def np_fourD_compound(U, S, VT):
    (bs, c, m, k) = U.shape
    n = VT.shape[3]
    SS = np.zeros((bs, c, k, k))
    X = np.zeros((bs, c, m, n))
    for i in range(bs):
        for j in range(c):
            for p in range(k):
                SS[i][j][p][p] = S[i, j, p]  # 奇异值list形成矩阵

    for i in range(bs):
        for j in range(c):
            pX = np.dot(U[i, j, :, :], SS[i, j, :, :])
            pX = np.dot(pX, VT[i, j, :, :])
            X[i, j, :, :] = pX
    return X


def make_optimizer_class(cls):

    class DPOptimizerClass(cls):

        def __init__(self, l2_norm_clip, noise_multiplier, batch_size, device, *args, **kwargs):
            """
            Args:
                l2_norm_clip (float): An upper bound on the 2-norm of the gradient w.r.t. the model parameters
                noise_multiplier (float): The ratio between the clipping parameter and the std of noise applied
                batch_size (int): Number of examples per batch
            """
            self.l2_norm_clip = l2_norm_clip
            self.noise = Normal(0.0, noise_multiplier * l2_norm_clip / (batch_size ** 0.5))
            self.device = device
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

        def step(self, closure=None):
            # Calculate total gradient
            total_norm = 0
            for group in self.param_groups:
                for p in filter(lambda p: p.grad is not None, group['params']):
                    param_norm = p.grad.data.norm(2.)
                    total_norm += param_norm.item() ** 2.
                total_norm = total_norm ** (1. / 2.)

            # Calculate clipping coefficient, apply if nontrivial
            clip_coef = self.l2_norm_clip / (total_norm + 1e-6)
            if clip_coef < 1:
                for group in self.param_groups:
                    for p in filter(lambda p: p.grad is not None, group['params']):
                            p.grad.data.mul_(clip_coef)

            # Inject noise
            for group in self.param_groups:
                for p in filter(lambda p: p.grad is not None, group['params']):
                    p.grad.data.add_(self.noise.sample(p.grad.data.size()).to(self.device))

            super(DPOptimizerClass, self).step(closure)

    return DPOptimizerClass


DPSGD = make_optimizer_class(SGD)

