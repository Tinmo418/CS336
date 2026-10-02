import torch
from torch import nn
import math
from einops import rearrange
from cs336_basics.linear_embedding_module import Linear, Embedding
from cs336_basics.normalization import RMSNorm

# RMSNorm
class positionwise_feedforward(nn.Module):
    # 初始化 SwiGLU 前馈网络的三组线性投影。
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.W_1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W_2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W_3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    # SwiGLU
    # 通过 SiLU 门控和线性投影更新每个位置的特征。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur = self.W_1(x)
        silu = cur * torch.sigmoid(cur)
        ffn = self.W_2(silu * self.W_3(x))
        return ffn


#max_seq_len的含义是RoPE一次性处理多少个token
class RotaryPositionalEmbedding(nn.Module):
    # 预计算 RoPE 各位置的正弦和余弦表。
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        positions = torch.arange(max_seq_len, device=device)
        positions = positions[:, None]
        k = torch.arange(d_k//2, device=device)
        freqs = 1 / theta**(2*k/d_k)
        angles = positions * freqs
        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)

        self.register_buffer(
            "cos_table",
            cos_table,
            persistent=None
        )
        self.register_buffer(
            "sin_table",
            sin_table,
            persistent=False
        )

    # 根据位置对相邻特征维度进行旋转，加入位置信息。
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        new_even = even * cos - odd * sin
        new_odd = even * sin + odd * cos
        #因为在stack里多了一个维度 且存放在最后一个位置 然后需要最后用flatten把最后两个shape里的维度合并
        return torch.stack((new_even, new_odd), dim=-1).flatten(-2, -1)


# 沿指定维度计算数值稳定的 softmax 概率。
def softmax(x: torch.Tensor, dim: int):
    c = torch.max(x, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(x-c)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)
    

# 根据 Q、K 计算注意力权重，并在可选掩码约束下加权汇总 V。
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    dk = Q.shape[-1]
    score = Q @ K.mT
    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    return softmax(score / math.sqrt(dk), dim=-1) @ V


class multihead_self_attention(nn.Module):
    # 初始化多头注意力的 Q、K、V、输出投影和 RoPE。
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0
        self.dk = self.dv = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        #这里其实本质是d_model->d_model 只是到了forward部分再把后面的这个d_model解释为h*d
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta, self.dk, max_seq_len, device=device)
        
    # 结合 RoPE 与因果掩码计算多头自注意力，并合并各头输出。
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        #保留上三角矩阵 取反 因为按照之前的逻辑是有True才给标记为-inf 而这里应该得是值为False给标记
        mask  = ~torch.triu(mask, diagonal=1)

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = rearrange(
            Q,
            "... seq_len (num_heads dk) -> ... num_heads seq_len dk",
            num_heads = self.num_heads,
            dk = self.dk
        )
        K = rearrange(
            K,
            "... seq_len (num_heads dk) -> ... num_heads seq_len dk",
            num_heads = self.num_heads,
            dk = self.dk
        )
        V = rearrange(
            V,
            "... seq_len (num_heads dv) -> ... num_heads seq_len dv",
            num_heads = self.num_heads,
            dv = self.dv
        )
        
        #V不需要经过RoPE 这里[:,None,:]是使(batch,seq_len)->(batch,1,seq_len) 使其可以与Q和K进行rope
        Q = self.rope(Q, token_positions[...,None,:])
        K = self.rope(K, token_positions[...,None,:])

        cur = scaled_dot_product_attention(Q, K, V, mask)
        merged_heads = rearrange(
            cur,
            "... num_heads seq_len dv -> ... seq_len (num_heads dv)"
        )
        return self.W_O(merged_heads)
        
        

class TransformerBlock(nn.Module):
    # 组装预归一化、因果自注意力和 SwiGLU 前馈模块。
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        eps: float = 1e-5

        self.rmsnorm1 = RMSNorm(d_model, eps, device, dtype)
        self.rmsnorm2 = RMSNorm(d_model, eps, device, dtype)
        self.mhsa = multihead_self_attention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.swiglu = positionwise_feedforward(d_model, d_ff, device, dtype)


    # 依次执行注意力与前馈计算，并分别保留残差连接。
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        y1 = self.mhsa(self.rmsnorm1(x), token_positions) + x
        y2 = self.swiglu(self.rmsnorm2(y1)) + y1
        return y2


class TransformerLM(nn.Module):
    # 组装 embedding、多层 Transformer、最终归一化和词表输出层。
    def __init__(self, vocab_size: int, context_len: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float, device=None, dtype=None):
        super().__init__()
        eps: float = 1e-5
        self.num_layers = num_layers

        #context_len就是原本的max_seq_len
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_len, device, dtype) for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model, eps, device, dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)
        

    # 将 token ID 序列转换为每个位置预测下一个 token 的 logits。
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_embedded = self.embedding(token_ids)
        seq_len = token_embedded.shape[-2]
        batch = token_embedded.shape[0]
        positions = torch.arange(seq_len, device=token_embedded.device).unsqueeze(0)
        token_positions = positions.expand(batch, -1)

        for i in range(self.num_layers):
            token_embedded = self.layers[i](token_embedded, token_positions)

        logits = self.linear(self.rmsnorm(token_embedded))
        return logits
