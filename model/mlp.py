"""Transformer FFN：GELU-MLP（GPT-2）或 SwiGLU（LLaMA 系）。"""

import torch.nn as nn
import torch.nn.functional as F


def resolve_ffn_hidden(n_embd: int, mlp_hidden_dim: int | None, ffn_type: str) -> int:
    """GELU 用 4*n_embd；SwiGLU 用约 2/3 宽度以控制参数量接近 GPT-2 FFN。"""
    h = mlp_hidden_dim if mlp_hidden_dim is not None else 4 * n_embd
    if ffn_type == "swiglu":
        return max(64, int(2 * h / 3))
    return h


class MLP(nn.Module):
    def __init__(self, n_embd: int, mlp_hidden_dim: int, dropout: float, bias: bool = True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, mlp_hidden_dim, bias=bias)
        self.c_proj = nn.Linear(mlp_hidden_dim, n_embd, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w3(F.silu(self.w1(x)) * self.w2(x))
        return self.dropout(x)


def build_ffn(
    ffn_type: str,
    n_embd: int,
    mlp_hidden_dim: int | None,
    dropout: float,
    bias: bool = True,
) -> nn.Module:
    hidden = resolve_ffn_hidden(n_embd, mlp_hidden_dim, ffn_type)
    if ffn_type == "gelu":
        return MLP(n_embd, hidden, dropout, bias=bias)
    if ffn_type == "swiglu":
        return SwiGLUFFN(n_embd, hidden, dropout, bias=bias)
    raise ValueError(f"未知 ffn_type={ffn_type!r}，可选 gelu | swiglu")
