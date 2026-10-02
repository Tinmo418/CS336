import torch
from torch import nn
import math

class RMSNorm(nn.Module):
    # 初始化 RMSNorm 的可学习缩放参数和数值稳定项。
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        matrix = torch.ones(
            d_model,
            device=device,
            dtype=dtype
        )
        self.weight = nn.Parameter(matrix)
        self.eps = eps

    # 沿最后一维进行 RMS 归一化，并恢复输入的数据类型。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x / (x.pow(2).mean(dim=-1, keepdim=True)+self.eps).sqrt() * self.weight
        return result.to(in_dtype)
        
