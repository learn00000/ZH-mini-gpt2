"""模型超参数：训练脚本构造 TinyGPT(cfg) 时只改这里即可对齐 baseline。"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int  # 字表大小，与 tokenizer 一致
    block_size: int = 256  # 最大上下文长度 T；因果 mask 按它定
    n_embd: int = 512  # 模型宽度 C；每头维度 = n_embd // n_head
    n_layer: int = 8  # TransformerBlock 堆叠层数
    n_head: int = 8  # 注意力头数
    dropout: float = 0.1  # 嵌入 / MLP / 注意力残差投影
    attn_dropout: float | None = None  # None 时与 dropout 相同；B2 可设为 0.2
    bias: bool = True  # Linear / LayerNorm 是否带 bias（与 GPT-2 一致可全开）
    mlp_hidden_dim: int | None = None  # None 时使用 4 * n_embd（与 GPT-2 FFN 比例一致）
    # 位置编码：wpe=baseline；rope=实验 A1（勿与 attn_window 同开）
    pos_encoding: str = "wpe"
    rope_theta: float = 10000.0
    # 滑动窗口注意力：None/0=全长；W>0 时每个 token 只看最近 W 个位置（实验 A2，仍用 wpe）
    attn_window: int | None = None
    # 实验 B1：softmax 前后在头维做可学习混合（建议与 rope 同开，勿与 attn_window 同开）
    talking_heads: bool = False
    # 骨干组件（LLaMA 系常见配方，与注意力算子正交）
    norm_type: str = "layernorm"  # layernorm | rmsnorm
    ffn_type: str = "gelu"  # gelu | swiglu
