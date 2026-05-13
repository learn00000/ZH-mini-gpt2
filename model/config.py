"""模型超参数：训练脚本构造 TinyGPT(cfg) 时只改这里即可对齐 baseline。"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int  # 字表大小，与 tokenizer 一致
    block_size: int = 256  # 最大上下文长度 T；wpe 与因果 mask 都按它定
    n_embd: int = 384  # 模型宽度 C；每头维度 = n_embd // n_head
    n_layer: int = 6  # TransformerBlock 堆叠层数
    n_head: int = 4  # 注意力头数
    dropout: float = 0.1
    bias: bool = True  # Linear / LayerNorm 是否带 bias（与 GPT-2 一致可全开）
    mlp_hidden_dim: int | None = None  # None 时使用 4 * n_embd（与 GPT-2 FFN 比例一致）
