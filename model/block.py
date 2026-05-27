"""单层 Transformer：Pre-LN + 残差。主路径维度始终是 (B, T, n_embd)。"""

import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import build_ffn
from .norm import build_norm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        mlp_hidden_dim: int,
        dropout: float,
        attn_dropout: float | None = None,
        bias: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        attn_window: int | None = None,
        talking_heads: bool = False,
        norm_type: str = "layernorm",
        ffn_type: str = "gelu",
    ):
        super().__init__()
        self.ln_1 = build_norm(norm_type, n_embd, bias=bias)
        self.attn = CausalSelfAttention(
            n_embd,
            n_head,
            block_size,
            dropout,
            attn_dropout=attn_dropout,
            resid_dropout=dropout,
            bias=bias,
            use_rope=use_rope,
            rope_theta=rope_theta,
            attn_window=attn_window,
            talking_heads=talking_heads,
        )
        self.ln_2 = build_norm(norm_type, n_embd, bias=bias)
        self.mlp = build_ffn(ffn_type, n_embd, mlp_hidden_dim, dropout, bias=bias)

    def forward(self, x):
        # Pre-LN：先 Norm 再子层，再与残差相加
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
