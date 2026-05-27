"""Rotary Position Embedding (RoPE)，用于 Q/K，替代绝对位置嵌入 wpe。"""

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, nh, T, hd); cos/sin: (T, hd)"""
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE 需要偶数 head_dim，得到 {head_dim}")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, hd)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-2)
        cos = self.cos_cached[:T].view(1, 1, T, -1)
        sin = self.sin_cached[:T].view(1, 1, T, -1)
        return apply_rotary_pos_emb(x, cos, sin)
