"""Transformer 里的 FFN：对每个位置独立做 256->1024->256，与 GPT-2 一致用 GELU。"""

import torch.nn as nn


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
