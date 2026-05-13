"""字级因果语言模型：token/位置嵌入 -> n_layer 个 Block -> LayerNorm -> 词表 logits。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .config import GPTConfig


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        mlp_h = config.mlp_hidden_dim if config.mlp_hidden_dim is not None else 4 * config.n_embd

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token 嵌入
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # 可学习绝对位置嵌入
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    mlp_hidden_dim=mlp_h,
                    dropout=config.dropout,
                    bias=config.bias,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # 与 wte 共享权重，省参且常见做法

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if idx.dim() != 2:
            raise ValueError(f"idx must be 2D (B, T), got shape {tuple(idx.shape)}")
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length T={T} exceeds block_size={self.config.block_size}. "
                "Truncate or increase block_size in config."
            )

        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd)，与 tok_emb 广播相加
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 与 nanoGPT 相同：展平后 CE；训练时 targets 需与 logits 对齐（常配合移位标签）
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            # 超长上下文只保留最后 block_size 个 token 再 forward
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # 只用最后一个位置的下一 token 分布

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
