import torch
from cs336_basics.tokenizer import (
    Tokenizer
)

from cs336_basics.transformer_modules import (
    softmax,
    TransformerLM
)


# 不叫decoding是为了和tokenizer里的 decode 区别开
# 当前版本会返回新增内容与原本的prompt
# 按温度和 top-p 逐步采样，遇到结束 token 或长度上限后返回 prompt 与续写 ID。
def generate(model: TransformerLM, prompt: list[int], special_token_id: int, temperature: float, maximum_tokens: int, top_p: float, context_length: int = 256) -> list[int] :
    model.eval()
    device = next(model.parameters()).device
    generated_ids = list(prompt)

    with torch.no_grad():
        for _ in range(maximum_tokens):
            current_context = generated_ids[-context_length:]
            input_ids = torch.tensor(
                current_context,
                dtype=torch.long,
                device=device
            ).unsqueeze(0) # 这里需要unsqueeze(0)是因为此时input_ids.shape==(seq_len,)而TransformerLM需要(batch_size, seq_len)

            logits = model(input_ids)

            # 这里[:,-1,:]是因为只需要取序列的最尾端去预测 不然相当于我已经输入了x0 x1 x2 x3却还需要根据x0去预测x1 根据x0和x1去预测x2...
            next_token_logits = logits[:,-1,:]
            probabilities = softmax(next_token_logits / temperature, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)

            # 这里cumsum可以一边求和一边记录stpes
            cumulative_p = torch.cumsum(sorted_probs, dim=-1)

            # 这里的返回结果是一个True/False的表 < top_p 判定为T
            mask = (cumulative_p - sorted_probs) < top_p
            filtered_probs = sorted_probs.masked_fill(~mask, 0.0)

            # 归一化
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            
            # 其实multinomial本身就是根据权重去选择下标的 反而上面没必要归一化 但归一了也无妨
            # num_samples=1 表示一次性取一个下标
            pos = torch.multinomial(filtered_probs, num_samples=1)

            # 当需要沿着某个特定维度取值时需要用到gather
            next_token_id = sorted_ids.gather(
                dim=-1,
                index=pos
            ).item() # item把tensor类型转换为int类型(虽然都是id)

            if next_token_id != special_token_id:
                generated_ids.append(next_token_id)
            else:
                break
          
    return generated_ids
