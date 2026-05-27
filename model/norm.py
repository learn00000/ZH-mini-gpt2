"""归一化层：LayerNorm（GPT-2）与 RMSNorm（LLaMA 系）。"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def build_norm(norm_type: str, n_embd: int, bias: bool = True) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(n_embd, bias=bias)
    if norm_type == "rmsnorm":
        return RMSNorm(n_embd)
    raise ValueError(f"未知 norm_type={norm_type!r}，可选 layernorm | rmsnorm")
