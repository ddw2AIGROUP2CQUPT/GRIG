"""
暂时用normalize 就好,其他功能当前任务无需使用,2022.10.21
"""
import random
import numpy as np
import torch
class Normalize:
    """
    normalize the graph nodes' feats

    Args：
        mean,std ： the mean and std of the  training dataset using for normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, nodes=torch.Tensor):
        assert isinstance(nodes, torch.Tensor)
        return nodes.sub_(self.mean).div_(self.std)
