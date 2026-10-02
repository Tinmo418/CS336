import torch
import numpy as np
from torch import nn


# 随机抽取连续 token 片段，返回输入和向后错一位的预测标签。
def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_list = []
    targets_list = []
    
    start = np.random.randint(0, len(x)-context_length, size=batch_size) #size表示生成多少个随机数
    for s in start:
        inputs_list.append(x[s:s+context_length])
        targets_list.append(x[s+1:s+1+context_length])

    inputs = torch.as_tensor(np.stack(inputs_list), dtype=torch.long, device=device)
    targets = torch.as_tensor(np.stack(targets_list), dtype=torch.long, device=device)
    return inputs, targets

# 保存模型权重、优化器状态和当前训练步数。
def save_checkpoint(model, optimizer, iteration, out):
    #iteration表示循环次数 且这里用不到它的具体数值 只是作记录
    #该函数不需要return值 只是保存到out指定的文件里
    
    #我一开始想的是用tuple保存 但感觉这个不太好 推荐用法是用dict
    #torch.save((model.state_dict(), optimizer.state_dict(), iteration), out)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


# 恢复模型与优化器状态，并返回保存时的训练步数。
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]



