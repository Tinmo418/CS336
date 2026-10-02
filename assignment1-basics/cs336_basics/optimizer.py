import torch
import math

class AdamW(torch.optim.Optimizer):
    # 注册待优化参数及 AdamW 的默认超参数。
    def __init__(self, params, lr=1e-3, betas=(0.9,0.95), eps=1e-8, weight_decay=0.0):
        #PyTorch 父类 Optimizer 的构造函数要求第二个参数 defaults 是一个字典
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    # 根据梯度更新一阶、二阶矩估计，并执行参数更新与权重衰减。
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                #state: 保存当前参数 p 对应的 AdamW 历史状态：更新步数 t，以及一阶、二阶矩估计 m 和 v
                state = self.state[p]
                if not state:
                    state["t"] = 0 #t表示更新次数
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                
                t = state.get("t", 0)
                t = t + 1
                state["t"] = t

                beta1, beta2 = group["betas"]
                g = p.grad
                updated_lr = lr * math.sqrt((1-beta2**t)) / (1-beta1**t)
                p.data -= lr * weight_decay * p.data
                updated_m = beta1 * state["m"] + (1 - beta1) * g
                updated_v = beta2 * state["v"] + (1 - beta2) * (g**2)
                p.data -= updated_lr * updated_m / (updated_v.sqrt() + eps)

                state["m"] = updated_m
                state["v"] = updated_v
                
        return loss


# 根据当前 iteration 计算这一轮应该使用的学习率
# iteration: 当前训练到第几步
# lr_max / lr_min: 学习率上升阶段最终达到的最大值 / 余弦下降结束后保留的最小学习率
# warmup_iters: warmup 持续多少步
# cosine_cycle_iters: 余弦退火在第几步结束
def CosineAnnealing(iteration, lr_max, lr_min, warmup_iters, cosine_cycle_iters):
    if iteration < warmup_iters:
        lr_t = iteration / warmup_iters * lr_max
    elif iteration > cosine_cycle_iters:
        lr_t = lr_min
    else:
        lr_t = lr_min + 1/2 * (1 + math.cos((iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (lr_max - lr_min)
    return lr_t


# 检查并缩放 p.grad
# 按所有参数梯度的总 L2 范数进行缩放，限制梯度大小。
def GradientClipping(params, max_norm, eps=1e-6):
    params = list(params)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    
    total = 0
    for p in params:
        if p.grad is not None:
            total += torch.sum(p.grad**2)

    G = torch.sqrt(total)

    if G >= max_norm:
        scale = max_norm / (G + eps)
        for g in grads:
            g.mul_(scale)
