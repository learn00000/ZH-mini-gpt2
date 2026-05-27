"""单层因果自注意力：输入 (B,T,C)，输出 (B,T,C)。C=n_embd，多头在 C 维上切分。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


def _mix_heads_pre(att: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """(B, H, Q, K) × (H, G) → (B, G, Q, K)，用 matmul 替代 einsum 以减开销。"""
    return torch.matmul(att.permute(0, 2, 3, 1), mix).permute(0, 3, 1, 2)


def _mix_heads_post(att: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """(B, G, Q, K) × (G, H) → (B, H, Q, K)"""
    return torch.matmul(att.permute(0, 2, 3, 1), mix).permute(0, 3, 1, 2)


def _build_causal_mask(block_size: int, attn_window: int | None) -> torch.Tensor:
    """(1, 1, T, T)：下三角因果；若 attn_window 给定，再限制每个 query 只看最近 W 个 key。"""
    causal = torch.tril(torch.ones(block_size, block_size))
    if attn_window is None or attn_window <= 0 or attn_window >= block_size:
        return causal.view(1, 1, block_size, block_size)
    q = torch.arange(block_size).view(-1, 1)
    k = torch.arange(block_size).view(1, -1)
    dist = q - k
    windowed = (dist >= 0) & (dist < attn_window)
    return (causal * windowed.float()).view(1, 1, block_size, block_size)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        attn_dropout: float | None = None,
        resid_dropout: float | None = None,
        bias: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        attn_window: int | None = None,
        talking_heads: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.use_rope = use_rope

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        ad = dropout if attn_dropout is None else attn_dropout
        rd = dropout if resid_dropout is None else resid_dropout
        self.attn_dropout = nn.Dropout(ad)
        self.resid_dropout = nn.Dropout(rd)
        # 为可视化脚本保留最近一次 forward 的注意力权重（训练时保持 False）
        self.record_enabled: bool = False
        self.last_attn: torch.Tensor | None = None  # (B, nh, T, T)，softmax 之后

        self.attn_window = attn_window
        self.register_buffer("bias", _build_causal_mask(block_size, attn_window))
        self.rope: RotaryEmbedding | None = None
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, block_size, theta=rope_theta)
        self.talking_heads = talking_heads
        if talking_heads:
            # 初始化为近似恒等，训练初期行为接近标准多头
            self.pre_head_mix = nn.Parameter(torch.eye(n_head))
            self.post_head_mix = nn.Parameter(torch.eye(n_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)  # (B, T, 3*C) 一次线性再拆成 q,k,v
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # (B, T, nh, hs) -> (B, nh, T, hs)，便于 batched matmul
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q = self.rope.apply(q)
            k = self.rope.apply(k)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, T)
        if self.talking_heads:
            # 先在头维混合 logits，再施加因果/窗口 mask（避免 -inf 线性组合出 NaN）
            att = _mix_heads_pre(att, self.pre_head_mix)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = _mix_heads_post(att, self.post_head_mix)
            att = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        else:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
        if self.record_enabled:
            self.last_attn = att.detach()
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
