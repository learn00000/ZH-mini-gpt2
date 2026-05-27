"""字级因果语言模型：token/位置嵌入 -> n_layer 个 Block -> LayerNorm -> 词表 logits。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .config import GPTConfig
from .norm import build_norm


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        mlp_h = config.mlp_hidden_dim if config.mlp_hidden_dim is not None else 4 * config.n_embd

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token 嵌入
        self.use_rope = config.pos_encoding == "rope"
        aw = config.attn_window
        self.attn_window = None if aw is None or aw <= 0 else int(aw)
        if self.use_rope and self.attn_window is not None:
            raise ValueError("A1(RoPE) 与 A2(滑动窗口) 勿同时开启；请 pos_encoding=wpe 且 --attn_window W")
        if config.talking_heads and self.attn_window is not None:
            raise ValueError("B1(Talking-Heads) 与 A2(滑动窗口) 勿同时开启")
        if self.use_rope:
            self.wpe = None
        else:
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
                    attn_dropout=config.attn_dropout,
                    bias=config.bias,
                    use_rope=self.use_rope,
                    rope_theta=config.rope_theta,
                    attn_window=self.attn_window,
                    talking_heads=config.talking_heads,
                    norm_type=config.norm_type,
                    ffn_type=config.ffn_type,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = build_norm(config.norm_type, config.n_embd, bias=config.bias)

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
        if self.wpe is not None:
            x = self.drop(tok_emb + self.wpe(pos))
        else:
            x = self.drop(tok_emb)

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
    def forward_collect_attentions(self, idx: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        单次 teacher-forcing forward，收集每层因果注意力（softmax 后、dropout 前）。
        返回 logits 与 list[attn]，attn 形状为 (B, n_head, T, T)。
        """
        was_training = self.training
        self.eval()
        for block in self.blocks:
            block.attn.record_enabled = True
            block.attn.last_attn = None

        logits, _ = self(idx, targets=None)

        attns: list[torch.Tensor] = []
        for block in self.blocks:
            a = block.attn.last_attn
            if a is None:
                raise RuntimeError("注意力未记录，请检查 CausalSelfAttention.record_enabled")
            attns.append(a)

        for block in self.blocks:
            block.attn.record_enabled = False

        if was_training:
            self.train()
        return logits, attns

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
