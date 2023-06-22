# coding:utf-8


import torch


def kl_loss4gp(p, q):
    """
    https://spaces.ac.cn/archives/9039

    f = sigmoid(x)
    KL_divergence = (f(p) - f(q)) * (p - q)
    """
    return ((torch.sigmoid(p) - torch.sigmoid(q)) * (p - q)).mean()





