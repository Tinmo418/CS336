import torch
from torch import nn
import math

class Linear(nn.Module):
    # 初始化无偏置线性层的权重矩阵。
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        k = math.sqrt(2 / (in_features+out_features))
        tensor = torch.empty(
            (out_features, in_features),
            device = device,
            dtype = dtype
        )
        self.weight = nn.Parameter(tensor)
        torch.nn.init.trunc_normal_(
            self.weight,
            #均值
            mean = 0.0,
            #标准差
            std = k,
            #保存范围
            a = -3*k,
            b = 3*k
        )

    # 对输入最后一维执行线性投影。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    # 初始化 token ID 对应的可学习向量表。
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        matrix = torch.empty(
            (num_embeddings, embedding_dim),
            device = device,
            dtype = dtype
        )

        self.weight = nn.Parameter(matrix)
        torch.nn.init.trunc_normal_(
            self.weight
        )

    # 根据 token ID 查找对应的 embedding 向量。
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
