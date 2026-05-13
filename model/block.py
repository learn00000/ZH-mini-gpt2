"""单层 Transformer：Pre-LN + 残差。主路径维度始终是 (B, T, n_embd)。"""

import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        mlp_hidden_dim: int,
        dropout: float,
        bias: bool = True,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, mlp_hidden_dim, dropout, bias=bias)

    def forward(self, x):
        # Pre-LN：先 Norm 再子层，再与残差相加
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
