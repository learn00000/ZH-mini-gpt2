"""
训练 / 调试前的人类可读摘要：
- 打印 GPTConfig、总参数量；可选逐参数与逐层 forward hook；
- compact=True 时只做一次 smoke forward，适合正式训练开跑前。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import GPTConfig


@torch.no_grad()
def print_model_summary(
    model: nn.Module,
    *,
    config: GPTConfig | None = None,
    batch_size: int = 2,
    seq_len: int = 16,
    device: torch.device | None = None,
    compact: bool = False,
) -> None:
    """
    打印：配置、总参数量；非 compact 时再打印每个参数张量与各子模块 forward 输出形状。
    正式训练建议 compact=True。
    """
    if device is None:
        device = next(model.parameters()).device

    cfg = config if config is not None else getattr(model, "config", None)
    if cfg is not None:
        print("=== GPTConfig ===")
        print(cfg)
        print()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=== Parameters ===")
    print(f"total params: {total:,}  trainable: {trainable:,}")
    print()

    if compact:
        vocab_size = cfg.vocab_size if cfg is not None else int(
            getattr(getattr(model, "config", None), "vocab_size", 1000)
        )
        model.eval()
        idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits, _ = model(idx, targets=None)
        print(
            f"=== Smoke forward ===  logits {tuple(logits.shape)}  (compact：已省略逐参数与逐层 hook)"
        )
        print()
        return

    print("=== Parameter tensors (name, shape, #elements) ===")
    for name, p in model.named_parameters():
        print(f"  {name:50s}  {tuple(p.shape)!s:28s}  {p.numel():,}")
    print()

    vocab_size = cfg.vocab_size if cfg is not None else int(getattr(getattr(model, "config", None), "vocab_size", 1000))
    model.eval()
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    forward_shapes: list[tuple[str, str]] = []

    def make_hook(mod_name: str):
        def hook(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                forward_shapes.append((mod_name, str(tuple(out.shape))))
            elif isinstance(out, tuple):
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        forward_shapes.append((f"{mod_name}[{i}]", str(tuple(o.shape))))

        return hook

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        _ = model(idx, targets=None)
    finally:
        for h in handles:
            h.remove()

    print(f"=== Forward tensor shapes (dummy idx: batch={batch_size}, T={seq_len}) ===")
    for name, shape in forward_shapes:
        print(f"  {name:50s}  ->  {shape}")
    print()

