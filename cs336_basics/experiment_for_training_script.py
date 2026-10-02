import numpy as np
import torch
from pathlib import Path


from cs336_basics.training_loop import (
    data_loading
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
seed = 42

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
validation_batches = 20 # 得到一次有效的 validation_loss 需要经过多少次的 forward
#训练总token数量(4kw) = num_iterations * batch_size * context_length
#这里num_iterations大约为19532 表示训练总共更新多少次
num_iterations = 1000
#推荐max_norm = 1.0 表示每一次更新前允许梯度最大有多大
max_norm = 1.0

lr_max = 5e-3
lr_min = 0.01 * lr_max
warmup_iters = int(0.02 * num_iterations)
cosine_cycle_iters = num_iterations

betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1


print_interval = 20
validation_interval = 100

# checkpoint 的路径
checkpoint_dir = Path(f"checkpoints/tinystories_lr{lr_max:.0e}_bs{batch_size}")

def evaluate_validation_loss(
    model: torch.nn.Module,
    validation_ids: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    validation_batches: int,
) -> float:
    validation_losses = []

    # 表明是对模型做评估
    model.eval() 
    # 表明不需要计算梯度 减少内存占用
    with torch.no_grad():
        for _ in range(validation_batches):
            inputs, targets = data_loading(validation_ids, batch_size, context_length, device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            validation_losses.append(loss.item())

    # 标记为恢复到训练模式 与eval相对
    model.train() 

    return sum(validation_losses) / len(validation_losses)



def main():
    np.random.seed(seed)
    torch.manual_seed(seed)

    token_ids = np.memmap("models/tinystories_ids.bin", dtype=np.uint16, mode="r")
    validation_ids = np.memmap("models/tinystories_valid_ids.bin", dtype=np.uint16, mode="r")
    
    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device, dtype)
    optimizer = AdamW(model.parameters(), lr=lr_max, betas=betas, eps=eps, weight_decay=weight_decay)


    for iteration in range(num_iterations):
        inputs, targets = data_loading(token_ids, batch_size, context_length, device)

        lr = CosineAnnealing(iteration, lr_max, lr_min, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        logits = model(inputs) 
        loss = cross_entropy(logits, targets)
        loss.backward() 
        GradientClipping(model.parameters(), max_norm) 
        optimizer.step() 

        if (iteration + 1) % print_interval == 0 or iteration + 1 == num_iterations:
            if (iteration + 1) % validation_interval == 0 or iteration + 1 == num_iterations:
                avg_loss = evaluate_validation_loss(model, validation_ids, batch_size, context_length, device, validation_batches)
                print(
                    f"step: {iteration+1}, "
                    f"train_loss: {loss:.4f}, "
                    f"validation_loss: {avg_loss:.4f}, "
                    f"lr: {lr:.6e}"
                )
            else:
                print(
                    f"step: {iteration + 1}, "
                    f"loss: {loss.item():.4f}, "
                    f"lr: {lr:.6e}"
                )

            

if __name__ == "__main__":
    main()
    