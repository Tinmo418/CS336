import torch
from cs336_basics.transformer_modules import (
    softmax
)

# 计算真实目标 token 的交叉熵，并对所有输入位置取平均。
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #logits.shape  == (batch, m, vocab_size)
    #targets.shape == (batch, m)
    #m==seq_len && m<=context_length
    assert logits.shape[:-1] == targets.shape
    correct_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))

    #这里可以直接用torch.logsumexp torch.logsumexp(logits, dim=-1)但还是手写实现 
    maximum = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - maximum
    shifted_exp = torch.exp(shifted_logits)
    sum_exp = shifted_exp.sum(dim=-1, keepdim=True)
    #上面把最大值减去防止溢出 计算完后还得加回来
    log_normalizer = torch.log(sum_exp) + maximum

    #先求两个求和号内部的-log_p mean()表示对所有元素求和再求平均
    #等价于per_token_loss.sum() / (m*batch_size)
    per_token_loss = (log_normalizer - correct_logits)
    return per_token_loss.mean()
    
    
    
    
