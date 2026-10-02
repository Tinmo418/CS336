import numpy as np
import torch
from pathlib import Path


from cs336_basics.training_loop import (
    data_loading,
    save_checkpoint
)
from cs336_basics.training_utils import (
    cross_entropy
)
from cs336_basics.transformer_modules import (
    TransformerLM
)
from cs336_basics.optimizer import (
    AdamW,
    CosineAnnealing,
    GradientClipping
)

vocab_size = 10_000
context_length = 256
d_model = 512
d_ff = 1344
theta = 10_000
num_layers = 4
num_heads = 16
device = 'mps' if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32
batch_size = 8 #这里没有固定值 可设置为4 8 16 32这种 越大steps越小
#训练总token数量(4kw) = num_iterations * batch_size * context_length
#这里num_iterations大约为19532 表示训练总共更新多少次
num_iterations = 19532
#推荐max_norm = 1.0 表示每一次更新前允许梯度最大有多大
max_norm = 1.0

lr_max = 5e-3
lr_min = 0.01 * lr_max
warmup_iters = int(0.02 * num_iterations)
cosine_cycle_iters = num_iterations

betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1

# 每隔多少次step保存一次 主要是储存吃不消啊 保存一次意思意思得了
checkpoint_interval = 10000
# 打印
print_interval = 500

# checkpoint 的路径
checkpoint_dir = Path(f"checkpoints/tinystories_lr{lr_max:.0e}_bs{batch_size}")



# 按配置训练语言模型，定期打印指标并保存 checkpoint。
def main():
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    token_ids = np.memmap("models/tinystories_ids.bin", dtype=np.uint16, mode="r")
    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device, dtype)
    optimizer = AdamW(model.parameters(), lr=lr_max, betas=betas, eps=eps, weight_decay=weight_decay)



    for iteration in range(num_iterations):
        inputs, targets = data_loading(token_ids, batch_size, context_length, device)

        # 需要更新每个interation里的学习率
        lr = CosineAnnealing(iteration, lr_max, lr_min, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 在每次反向传播之前先清空optimizer的上一次的梯度
        # optimizer本质是一个载体 只是初始化在循环之外 防止每次循环重新建立
        optimizer.zero_grad()
        logits = model(inputs) # forward
        loss = cross_entropy(logits, targets)
        loss.backward() # pythorch自带的 产生p.grad
        GradientClipping(model.parameters(), max_norm) # 检查并缩放p.gard
        optimizer.step() # 向参数更新p.grad

        if (iteration + 1) % checkpoint_interval == 0 or (iteration + 1) == num_iterations:
            out = checkpoint_dir / f"step_{iteration + 1}.pt"
            save_checkpoint(model, optimizer, iteration+1, out)

        if (iteration + 1) % print_interval == 0 or iteration + 1 == num_iterations:
            print(
                f"step: {iteration + 1}, "
                f"loss: {loss.item():.4f}, "
                f"lr: {lr:.6e}"
            )
 

if __name__ == "__main__":
    main()
    
