#!/usr/bin/env python3
"""
Mini GPT-2（中文、字级）训练入口。

【prepare 在做什么】
  把 JSONL（每行 {"text": "..."}）按分词器（字级 char 或 GPT-2 BPE）编成 token id；每条样本可加 <bos>/<eos>、
  末尾加 <eos>，再将各条顺序拼接（packed）写入 train.bin / valid.bin（uint16 memmap）。
  训练时随机切长度为 block_size 的窗口，不必把整份 JSONL 一次性读进内存。

【checkpoint】
  每次 train 在 checkpoint_dir 下新建时间戳子目录：best.pt、last.pt、run_info.json，
  loss_history.json（CE / PPL / BPC）、loss_curve.png；
  结束后默认用 best.pt 在 {data_root}/data/test.jsonl 上评测并写 test_report.json（--no_test_after_train 可关）。

【用法】
  python train.py prepare --split data [--prepare_force]
  python train.py train --split data --device cuda --max_steps 20000
  显存不足时可保持等效 batch：例如 --batch_size 16 --gradient_accumulation_steps 2

默认数据根目录 /root/autodl-tmp；本地可改 --data_root。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from model.config import GPTConfig
from model.gpt import TinyGPT
from model.summary import print_model_summary
from tokenizer import Tokenizer, load_tokenizer, vocab_size_of

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

META_NAME = "meta.json"
TRAIN_BIN = "train.bin"
VALID_BIN = "valid.bin"


@dataclass
class TrainConfig:
    """数据路径 + 与 GPTConfig 对齐的结构超参 + 优化器 / 日志。"""

    data_root: str = "/root/autodl-tmp"
    split: str = "minidata"
    tokenizer_dir: str = ""
    tokenizer_backend: str = "char"  # char | gpt2
    token_cache_dir: str = ""
    train_jsonl: str = "train.jsonl"
    valid_jsonl: str = "valid.jsonl"
    text_field: str = "text"

    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.0  # 0 表示与 dropout 相同；B2 用 0.2
    bias: bool = True
    pos_encoding: str = "wpe"  # wpe | rope（实验 A1，已完成勿与 A2 同开）
    rope_theta: float = 10000.0
    attn_window: int = 0  # 0=全长；>0 为 A2 滑动窗口 W（须 pos_encoding=wpe）
    talking_heads: bool = False  # B1：与 --rope 同开
    norm_type: str = "layernorm"  # layernorm | rmsnorm
    ffn_type: str = "gelu"  # gelu | swiglu

    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    lr_decay_ratio: float = 0.1
    max_steps: int = 20_000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    seed: int = 42
    device: str = "cuda"
    eval_interval: int = 200
    eval_iters: int = 200
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    resume: str = ""
    no_loss_plot: bool = False  # True 时不生成 loss_curve.png（仍写 loss_history.json）
    # 训练结束后在测试集上评估 best.pt（默认 {data_root}/data/test.jsonl）
    test_jsonl: str = ""  # 空则优先 {split}/test.jsonl，否则回退 data/test.jsonl
    no_test_after_train: bool = False  # True 则跳过测试集评估

    prepare_force: bool = False
    prepare_max_lines: Optional[int] = None

    def resolved_tokenizer_dir(self) -> str:
        if self.tokenizer_dir:
            return self.tokenizer_dir
        if self.tokenizer_backend == "gpt2":
            return os.path.join(self.data_root, "gpt2_tokenizer")
        return os.path.join(self.data_root, "char_tokenizer")

    def resolved_token_cache_dir(self) -> str:
        if self.token_cache_dir:
            return self.token_cache_dir
        if self.tokenizer_backend == "gpt2":
            return os.path.join(self.data_root, f"tokens_gpt2_{self.split}")
        return os.path.join(self.data_root, f"tokens_{self.split}")

    def dataset_dir(self) -> str:
        return os.path.join(self.data_root, self.split)

    def train_jsonl_path(self) -> str:
        return os.path.join(self.dataset_dir(), self.train_jsonl)

    def valid_jsonl_path(self) -> str:
        return os.path.join(self.dataset_dir(), self.valid_jsonl)

    def resolved_test_jsonl(self) -> str:
        if self.test_jsonl:
            return (
                self.test_jsonl
                if os.path.isabs(self.test_jsonl)
                else os.path.join(self.data_root, self.test_jsonl)
            )
        split_test = os.path.join(self.dataset_dir(), "test.jsonl")
        if os.path.isfile(split_test):
            return split_test
        return os.path.join(self.data_root, "data", "test.jsonl")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_LN2 = math.log(2.0)


def _ce_to_ppl(ce: float) -> float:
    """PyTorch CE 为 nat/token；PPL = exp(CE)。"""
    if ce >= 80.0:
        return float("inf")
    return math.exp(ce)


def _ce_to_bpc(ce: float) -> float:
    """字级 LM：bits per character = CE(nat) / ln(2)。"""
    return ce / _LN2


def _get_lr(step: int, *, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    decay_steps = max_steps - warmup_steps
    t = (step - warmup_steps) / max(decay_steps, 1)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# 语料：JSONL -> memmap
# ---------------------------------------------------------------------------


def _encode_lines(
    path: str,
    tok: Tokenizer,
    text_field: str,
    dtype: np.dtype,
    max_lines: Optional[int],
) -> Iterator[np.ndarray]:
    chunk: list[int] = []
    chunk_cap = 1_048_576
    n_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and n_lines >= max_lines:
                break
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "")
            if not text:
                continue
            chunk.extend(tok.encode(text, add_bos=True, add_eos=True))
            while len(chunk) >= chunk_cap:
                yield np.asarray(chunk[:chunk_cap], dtype=dtype)
                chunk = chunk[chunk_cap:]
    if chunk:
        yield np.asarray(chunk, dtype=dtype)


def _build_token_memmap(
    jsonl_path: str,
    out_bin_path: str,
    tok: Tokenizer,
    *,
    text_field: str = "text",
    dtype: np.dtype = np.uint16,
    max_lines: Optional[int] = None,
) -> int:
    if vocab_size_of(tok) >= 65536:
        raise ValueError("词表 >= 65536 请改用 uint32")
    total = 0
    os.makedirs(os.path.dirname(out_bin_path) or ".", exist_ok=True)
    with open(out_bin_path, "wb") as out:
        for arr in _encode_lines(jsonl_path, tok, text_field, dtype, max_lines):
            out.write(arr.tobytes())
            total += int(arr.shape[0])
    return total


def _write_meta(cache_dir: str, meta: dict) -> None:
    with open(os.path.join(cache_dir, META_NAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _read_meta(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, META_NAME), "r", encoding="utf-8") as f:
        return json.load(f)


class MemmapTokenCorpus:
    def __init__(self, bin_path: str, dtype: np.dtype = np.uint16):
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(bin_path)
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.length = int(self.data.shape[0])

    def get_batch(
        self, block_size: int, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = self.length - block_size - 1
        if max_start < 0:
            raise ValueError(f"token 数 {self.length} < block_size+1={block_size + 1}")
        ix = torch.randint(max_start + 1, (batch_size,))
        x = torch.zeros(batch_size, block_size, dtype=torch.long)
        y = torch.zeros(batch_size, block_size, dtype=torch.long)
        for b, i in enumerate(ix.tolist()):
            x[b] = torch.from_numpy(np.asarray(self.data[i : i + block_size], dtype=np.int64))
            y[b] = torch.from_numpy(np.asarray(self.data[i + 1 : i + 1 + block_size], dtype=np.int64))
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# prepare / train
# ---------------------------------------------------------------------------


def prepare(tc: TrainConfig) -> None:
    tok_dir = tc.resolved_tokenizer_dir()
    cache_dir = tc.resolved_token_cache_dir()
    train_path = tc.train_jsonl_path()
    valid_path = tc.valid_jsonl_path()

    if not os.path.isdir(tok_dir):
        raise FileNotFoundError(f"tokenizer 目录不存在: {tok_dir}")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"训练 jsonl 不存在: {train_path}")

    os.makedirs(cache_dir, exist_ok=True)
    train_bin = os.path.join(cache_dir, TRAIN_BIN)
    valid_bin = os.path.join(cache_dir, VALID_BIN)

    if not tc.prepare_force and os.path.isfile(train_bin):
        print(f"已存在 {train_bin}，跳过（使用 --prepare_force 覆盖）")
        return

    tok = load_tokenizer(tok_dir)
    vocab_size = vocab_size_of(tok)
    ml = tc.prepare_max_lines

    print(f"编码训练集 -> {train_bin} ...")
    n_train = _build_token_memmap(train_path, train_bin, tok, text_field=tc.text_field, max_lines=ml)
    n_valid = 0
    if os.path.isfile(valid_path):
        print(f"编码验证集 -> {valid_bin} ...")
        n_valid = _build_token_memmap(valid_path, valid_bin, tok, text_field=tc.text_field, max_lines=ml)
    else:
        print(f"未找到 {valid_path}，跳过 valid.bin")

    meta = {
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "train_tokens": n_train,
        "valid_tokens": n_valid,
        "train_jsonl": train_path,
        "valid_jsonl": valid_path if os.path.isfile(valid_path) else None,
        "tokenizer_dir": tok_dir,
        "tokenizer_backend": tc.tokenizer_backend,
        "text_field": tc.text_field,
        "prepare_max_lines": ml,
        "append_eos_per_line": True,
        "append_bos_per_line": True,
    }
    _write_meta(cache_dir, meta)
    print(f"完成：{os.path.join(cache_dir, META_NAME)}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if ml is not None:
        print(
            "\n【注意】使用了 --prepare_max_lines 仅作联调；正式训练请去掉该参数并 "
            "`python train.py prepare --split ... --prepare_force` 全量重编码。"
        )


@torch.no_grad()
def _estimate_loss(
    model: TinyGPT,
    corpus: MemmapTokenCorpus,
    tc: TrainConfig,
    device: torch.device,
    iters: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = corpus.get_batch(tc.block_size, tc.batch_size, device)
        _, loss = model(x, targets=y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _new_run_checkpoint_dir(parent: str) -> str:
    """在 parent 下新建唯一子目录（时间戳 + 纳秒后几位），用于本次训练所有产出。"""
    os.makedirs(parent, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns())[-8:]
    path = os.path.join(parent, name)
    os.makedirs(path, exist_ok=False)
    return path


def _write_run_info_json(run_dir: str, info: Dict[str, Any]) -> None:
    """单次写入：训练/模型摘要（无权重），固定文件名 run_info.json。"""
    path = os.path.join(run_dir, "run_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def _write_loss_history_json(run_dir: str, payload: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, "loss_history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _try_save_loss_curve_png(
    run_dir: str,
    train_steps: list[int],
    train_losses: list[float],
    val_steps: list[int],
    val_losses: list[float],
    train_at_val: list[float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip loss_curve.png (pip install matplotlib)")
        return

    if not train_steps and not val_steps:
        return

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    if train_steps:
        ax0.plot(
            train_steps,
            train_losses,
            label="train CE (single batch at log_interval)",
            color="tab:blue",
            alpha=0.85,
        )
    if val_steps:
        ax0.plot(
            val_steps,
            val_losses,
            "o-",
            label="val CE (mean over eval_iters batches)",
            color="tab:orange",
            markersize=4,
        )
    ax0.set_ylabel("cross-entropy")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.25)
    ax0.set_title("Loss: watch val vs train gap for overfitting")

    if val_steps and train_at_val and len(val_steps) == len(train_at_val):
        gap = [v - t for v, t in zip(val_losses, train_at_val)]
        ax1.plot(val_steps, gap, "o-", color="tab:red", markersize=4, label="val - train (same step)")
        ax1.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
        ax1.set_ylabel("val - train")
        ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlabel("optimizer step")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    out = os.path.join(run_dir, "loss_curve.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote loss curve: {out}")


def _save_ckpt(
    path: str,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float,
    tc: TrainConfig,
    *,
    val_loss: Optional[float] = None,
    train_loss: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    saved_at = datetime.now().isoformat(timespec="seconds")
    torch.save(
        {
            "saved_at": saved_at,
            "step": step,
            "best_val_loss": best_val,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gpt_config": asdict(model.config),
            "train_config": tc.to_dict(),
        },
        path,
    )


def _load_ckpt(path: str, model: TinyGPT, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[int, float]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"]), float(ckpt.get("best_val_loss", 1e9))


def _load_model_state_only(path: str, model: TinyGPT, device: torch.device) -> None:
    """仅从 checkpoint 加载 model 权重（用于测试集评估）。"""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)


@torch.no_grad()
def _evaluate_on_jsonl_sequential(
    model: TinyGPT,
    tok: Tokenizer,
    jsonl_path: str,
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
    text_field: str = "text",
    max_lines: Optional[int] = None,
) -> Dict[str, Any]:
    """
    将 jsonl 各行 text 编码为「<bos> + 正文 + <eos>」后顺序拼接为 token 流，按不重叠 block 切成 B 路并行窗口，
    累积 sum(loss * BT) / 总预测位置 得到 mean CE，再换算 PPL / BPC（与 prepare 的 packed+bos+eos 一致）。
    """
    model.eval()
    buf: list[int] = []
    sum_ce_weighted = 0.0
    n_positions = 0
    n_lines = 0

    def try_one_batch() -> bool:
        nonlocal buf, sum_ce_weighted, n_positions
        if len(buf) < block_size + 1:
            return False
        max_b = (len(buf) - 1) // block_size
        B = min(batch_size, max_b)
        if B < 1:
            return False
        x = torch.zeros(B, block_size, dtype=torch.long, device=device)
        y = torch.zeros(B, block_size, dtype=torch.long, device=device)
        for b in range(B):
            off = b * block_size
            x[b] = torch.tensor(buf[off : off + block_size], dtype=torch.long, device=device)
            y[b] = torch.tensor(buf[off + 1 : off + block_size + 1], dtype=torch.long, device=device)
        _, loss = model(x, targets=y)
        assert loss is not None
        sum_ce_weighted += loss.item() * (B * block_size)
        n_positions += B * block_size
        buf = buf[B * block_size :]
        return True

    def drain() -> None:
        while try_one_batch():
            pass

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and n_lines >= max_lines:
                break
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "")
            if not text:
                continue
            buf.extend(tok.encode(text, add_bos=True, add_eos=True))
            drain()
    drain()

    if n_positions <= 0:
        return {
            "test_jsonl": os.path.abspath(jsonl_path),
            "error": "no_tokens_evaluated",
            "mean_cross_entropy": None,
            "perplexity": None,
            "bpc": None,
        }
    mean_ce = sum_ce_weighted / n_positions
    return {
        "test_jsonl": os.path.abspath(jsonl_path),
        "num_pred_positions": n_positions,
        "mean_cross_entropy": mean_ce,
        "perplexity": _ce_to_ppl(mean_ce),
        "bpc": _ce_to_bpc(mean_ce),
        "eval_note": "Per-line: <bos> + text + <eos>, then concatenated; non-overlapping blocks; differs from per-document perplexity.",
    }


def train(tc: TrainConfig) -> None:
    torch.manual_seed(tc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tc.seed)

    if tc.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("警告：CUDA 不可用，改用 CPU。")
    else:
        device = torch.device(tc.device)

    cache_dir = tc.resolved_token_cache_dir()
    train_bin = os.path.join(cache_dir, TRAIN_BIN)
    valid_bin = os.path.join(cache_dir, VALID_BIN)
    meta_path = os.path.join(cache_dir, META_NAME)

    if not os.path.isfile(train_bin) or not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"未找到 token 缓存：{train_bin}\n请先：python train.py prepare --split {tc.split}"
        )

    meta = _read_meta(cache_dir)
    vocab_size = int(meta["vocab_size"])
    print(f"分词器: {meta.get('tokenizer_backend', tc.tokenizer_backend)}  vocab={vocab_size}")
    attn_w = tc.attn_window if tc.attn_window > 0 else None
    if tc.pos_encoding == "rope" and attn_w is not None:
        raise ValueError("A1(RoPE) 与 A2(滑动窗口) 不能同时开：请去掉 --rope 或 --attn_window/--sliding_window")
    if tc.talking_heads and attn_w is not None:
        raise ValueError("B1(Talking-Heads) 与 A2(滑动窗口) 不能同时开")
    if tc.talking_heads and tc.attn_dropout > 0 and tc.attn_dropout != tc.dropout:
        raise ValueError("B1 与 B2(attn_dropout) 请分两次训练，勿同开")

    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=tc.block_size,
        n_layer=tc.n_layer,
        n_head=tc.n_head,
        n_embd=tc.n_embd,
        dropout=tc.dropout,
        attn_dropout=(tc.attn_dropout if tc.attn_dropout > 0 else None),
        bias=tc.bias,
        pos_encoding=tc.pos_encoding,
        rope_theta=tc.rope_theta,
        attn_window=attn_w,
        talking_heads=tc.talking_heads,
        norm_type=tc.norm_type,
        ffn_type=tc.ffn_type,
    )
    print(f"骨干: norm={gpt_cfg.norm_type} ffn={gpt_cfg.ffn_type}")
    pe_note = f" (theta={gpt_cfg.rope_theta})" if gpt_cfg.pos_encoding == "rope" else ""
    print(f"位置编码: {gpt_cfg.pos_encoding}{pe_note}")
    if gpt_cfg.attn_dropout is not None and gpt_cfg.attn_dropout != gpt_cfg.dropout:
        print(f"注意力 dropout: {gpt_cfg.attn_dropout}（resid/MLP 仍为 {gpt_cfg.dropout}，实验 B2）")
    if gpt_cfg.talking_heads:
        print("注意力: Talking-Heads（实验 B1，softmax 前后头维混合）")
    if gpt_cfg.attn_window:
        print(f"注意力窗口: W={gpt_cfg.attn_window}（滑动局部因果，实验 A2）")
    elif not gpt_cfg.talking_heads:
        print("注意力: 标准多头（无 Talking-Heads）")
    model = TinyGPT(gpt_cfg).to(device)
    print_model_summary(
        model,
        config=gpt_cfg,
        batch_size=min(tc.batch_size, 4),
        seq_len=min(gpt_cfg.block_size, 64),
        device=device,
        compact=True,
    )

    train_corpus = MemmapTokenCorpus(train_bin)
    val_corpus = MemmapTokenCorpus(valid_bin) if os.path.isfile(valid_bin) else train_corpus
    print(
        f"train_tokens={train_corpus.length:,} val_tokens={val_corpus.length:,} "
        f"vocab={vocab_size} device={device}"
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        betas=(tc.beta1, tc.beta2),
        weight_decay=tc.weight_decay,
    )
    step = 0
    best_val = 1e9
    best_snap_step = 0
    if tc.resume and os.path.isfile(tc.resume):
        step, best_val = _load_ckpt(tc.resume, model, opt, device)
        best_snap_step = step
        print(f"恢复 checkpoint：step={step} best_val={best_val:.4f}")

    min_lr = tc.learning_rate * tc.lr_decay_ratio
    model.train()
    os.makedirs(tc.checkpoint_dir, exist_ok=True)
    run_dir = _new_run_checkpoint_dir(tc.checkpoint_dir)
    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"本次训练 checkpoint 目录：{os.path.abspath(run_dir)}")

    hist_train_steps: list[int] = []
    hist_train_losses: list[float] = []
    hist_train_ppl: list[Optional[float]] = []
    hist_train_bpc: list[float] = []
    hist_val_steps: list[int] = []
    hist_val_losses: list[float] = []
    hist_val_ppl: list[Optional[float]] = []
    hist_val_bpc: list[float] = []
    hist_train_at_val: list[float] = []

    t0 = time.time()
    last_val_loss: Optional[float] = None

    while step < tc.max_steps:
        lr = _get_lr(
            step,
            max_steps=tc.max_steps,
            warmup_steps=tc.warmup_steps,
            max_lr=tc.learning_rate,
            min_lr=min_lr,
        )
        for pg in opt.param_groups:
            pg["lr"] = lr

        accum = tc.gradient_accumulation_steps
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(accum):
            x, y = train_corpus.get_batch(tc.block_size, tc.batch_size, device)
            _, loss = model(x, targets=y)
            assert loss is not None
            (loss / accum).backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        step += 1

        if step % tc.log_interval == 0 or step == 1:
            tr_ppl = _ce_to_ppl(loss_accum)
            tr_bpc = _ce_to_bpc(loss_accum)
            tr_ppl_s = f"{tr_ppl:.2f}" if math.isfinite(tr_ppl) else "inf"
            print(
                f"step {step:6d}/{tc.max_steps}  lr {lr:.2e}  train_ce {loss_accum:.4f}  "
                f"ppl {tr_ppl_s}  bpc {tr_bpc:.4f}  ({time.time() - t0:.1f}s)"
            )
            hist_train_steps.append(step)
            hist_train_losses.append(loss_accum)
            hist_train_ppl.append(tr_ppl if math.isfinite(tr_ppl) else None)
            hist_train_bpc.append(tr_bpc)

        if step % tc.eval_interval == 0 or step == tc.max_steps:
            val_loss = _estimate_loss(model, val_corpus, tc, device, tc.eval_iters)
            va_ppl = _ce_to_ppl(val_loss)
            va_bpc = _ce_to_bpc(val_loss)
            va_ppl_s = f"{va_ppl:.2f}" if math.isfinite(va_ppl) else "inf"
            print(
                f"          >> val_ce {val_loss:.4f}  ppl {va_ppl_s}  bpc {va_bpc:.4f}"
            )
            last_val_loss = val_loss
            hist_val_steps.append(step)
            hist_val_losses.append(val_loss)
            hist_val_ppl.append(va_ppl if math.isfinite(va_ppl) else None)
            hist_val_bpc.append(va_bpc)
            hist_train_at_val.append(loss_accum)
            if val_loss < best_val:
                best_val = val_loss
                best_snap_step = step
                best_path = os.path.join(run_dir, "best.pt")
                _save_ckpt(
                    best_path,
                    model,
                    opt,
                    step,
                    best_val,
                    tc,
                    val_loss=val_loss,
                    train_loss=loss_accum,
                )
                print(f"          >> 保存 {best_path}")

    last_path = os.path.join(run_dir, "last.pt")
    _save_ckpt(
        last_path,
        model,
        opt,
        step,
        best_val,
        tc,
        val_loss=last_val_loss,
        train_loss=None,
    )
    finished_at = datetime.now().isoformat(timespec="seconds")

    min_val: Optional[float] = None
    min_val_step: Optional[int] = None
    if hist_val_losses:
        j = min(range(len(hist_val_losses)), key=lambda k: hist_val_losses[k])
        min_val = hist_val_losses[j]
        min_val_step = hist_val_steps[j]

    loss_payload: Dict[str, Any] = {
        "note": "CE in nats/token; PPL=exp(CE); BPC=CE/ln(2). Train: single batch at log points; val: mean over eval batches.",
        "train_log": {
            "step": hist_train_steps,
            "ce": hist_train_losses,
            "ppl": hist_train_ppl,
            "bpc": hist_train_bpc,
        },
        "val_log": {
            "step": hist_val_steps,
            "ce": hist_val_losses,
            "ppl": hist_val_ppl,
            "bpc": hist_val_bpc,
            "train_ce_same_step": hist_train_at_val,
        },
    }
    _write_loss_history_json(run_dir, loss_payload)
    print(f"已写 loss 历史：{os.path.join(run_dir, 'loss_history.json')}")

    if not tc.no_loss_plot:
        _try_save_loss_curve_png(
            run_dir,
            hist_train_steps,
            hist_train_losses,
            hist_val_steps,
            hist_val_losses,
            hist_train_at_val,
        )

    test_report: Optional[Dict[str, Any]] = None
    if not tc.no_test_after_train:
        best_pt = os.path.join(run_dir, "best.pt")
        test_path = tc.resolved_test_jsonl()
        if not os.path.isfile(best_pt):
            print("【测试集】未找到 best.pt，跳过。")
        elif not os.path.isfile(test_path):
            print(f"【测试集】未找到文件，跳过：{test_path}")
        else:
            print(f"【测试集】加载 best.pt，评测：{test_path}")
            tok_eval = load_tokenizer(tc.resolved_tokenizer_dir())
            eval_model = TinyGPT(gpt_cfg).to(device)
            _load_model_state_only(best_pt, eval_model, device)
            test_report = _evaluate_on_jsonl_sequential(
                eval_model,
                tok_eval,
                test_path,
                block_size=tc.block_size,
                batch_size=tc.batch_size,
                device=device,
                text_field=tc.text_field,
            )
            safe_report = dict(test_report)
            ppv = safe_report.get("perplexity")
            if isinstance(ppv, float) and not math.isfinite(ppv):
                safe_report["perplexity"] = None
            tp = os.path.join(run_dir, "test_report.json")
            with open(tp, "w", encoding="utf-8") as f:
                json.dump(safe_report, f, ensure_ascii=False, indent=2)
            print("========== 测试集（best.pt）==========")
            if safe_report.get("error"):
                print(f"  error: {safe_report['error']}")
            else:
                print(f"  pred_positions = {safe_report['num_pred_positions']}")
                print(f"  mean_ce          = {safe_report['mean_cross_entropy']:.6f}")
                ppx = safe_report.get("perplexity")
                pps = f"{ppx:.2f}" if isinstance(ppx, float) and ppx is not None and math.isfinite(ppx) else str(ppx)
                print(f"  ppl              = {pps}")
                print(f"  bpc              = {safe_report['bpc']:.6f}")
            print(f"  已写 {tp}")
            print("======================================")

    arts: Dict[str, str] = {
        "best": "best.pt",
        "last": "last.pt",
        "loss_history": "loss_history.json",
    }
    if not tc.no_loss_plot:
        arts["loss_curve"] = "loss_curve.png"
    if test_report is not None:
        arts["test_report"] = "test_report.json"

    run_info: Dict[str, Any] = {
        "run_dir": os.path.abspath(run_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "resumed_from": tc.resume if tc.resume else None,
        "gpt_config": asdict(model.config),
        "train_config": tc.to_dict(),
        "best_val_loss": best_val,
        "best_step": best_snap_step,
        "final_step": step,
        "final_train_loss_last_batch_ce": loss_accum,
        "final_train_ppl": _ce_to_ppl(loss_accum) if math.isfinite(_ce_to_ppl(loss_accum)) else None,
        "final_train_bpc": _ce_to_bpc(loss_accum),
        "last_measured_val_loss": last_val_loss,
        "last_measured_val_ppl": _ce_to_ppl(last_val_loss) if last_val_loss is not None and math.isfinite(_ce_to_ppl(last_val_loss)) else None,
        "last_measured_val_bpc": _ce_to_bpc(last_val_loss) if last_val_loss is not None else None,
        "min_val_loss": min_val,
        "min_val_loss_step": min_val_step,
        "test_on_best": test_report,
        "artifacts": arts,
    }
    run_info = {k: v for k, v in run_info.items() if v is not None}
    _write_run_info_json(run_dir, run_info)

    print("========== 训练结束 ==========")
    print(f"  final_step               = {step}")
    print(f"  末尾 train_ce（batch）   ≈ {loss_accum:.6f}")
    ftp = _ce_to_ppl(loss_accum)
    print(f"  末尾 train_ppl / bpc     = {ftp if math.isfinite(ftp) else 'inf'} / {_ce_to_bpc(loss_accum):.6f}")
    print(f"  最后一次 val_ce          = {last_val_loss}")
    if last_val_loss is not None:
        lvp = _ce_to_ppl(last_val_loss)
        print(
            f"  最后一次 val_ppl / bpc   = {lvp if math.isfinite(lvp) else 'inf'} / {_ce_to_bpc(last_val_loss):.6f}"
        )
    if min_val is not None and min_val_step is not None:
        mvp = _ce_to_ppl(min_val)
        print(
            f"  验证最低 val_ce / bpc    = {min_val:.6f} / {_ce_to_bpc(min_val):.6f}  (step {min_val_step}, ppl {mvp if math.isfinite(mvp) else 'inf'})"
        )
    print(f"  产出目录：{run_dir}")
    print("================================")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _apply_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp")
    p.add_argument("--split", type=str, default="data")
    p.add_argument("--tokenizer_dir", type=str, default="")
    p.add_argument(
        "--tokenizer",
        type=str,
        choices=("char", "gpt2"),
        default="char",
        help="char=字级词表；gpt2=OpenAI GPT-2 BPE（需先 setup_gpt2_tokenizer + prepare）",
    )
    p.add_argument("--token_cache_dir", type=str, default="")


def _cfg_from_ns(ns: argparse.Namespace) -> TrainConfig:
    names = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in vars(ns).items() if k in names})
    if getattr(ns, "rope", False):
        cfg.pos_encoding = "rope"
    if getattr(ns, "sliding_window", False) and cfg.attn_window <= 0:
        cfg.attn_window = 128
    if getattr(ns, "b2", False):
        cfg.attn_dropout = 0.2
        if cfg.pos_encoding != "rope":
            cfg.pos_encoding = "rope"
    backbone = getattr(ns, "backbone", "") or ""
    if backbone == "a1b1":
        cfg.pos_encoding = "rope"
        cfg.talking_heads = True
    elif backbone in ("a1b1_modern", "v2"):
        cfg.pos_encoding = "rope"
        cfg.talking_heads = True
        cfg.norm_type = "rmsnorm"
        cfg.ffn_type = "swiglu"
    if getattr(ns, "swiglu", False):
        cfg.ffn_type = "swiglu"
    if getattr(ns, "rmsnorm", False):
        cfg.norm_type = "rmsnorm"
    if hasattr(ns, "tokenizer") and ns.tokenizer:
        cfg.tokenizer_backend = ns.tokenizer
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini GPT-2 中文：prepare / train")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="JSONL -> train.bin / valid.bin (uint16 memmap)")
    _apply_common(p_prep)
    p_prep.add_argument("--text_field", type=str, default="text")
    p_prep.add_argument("--prepare_force", action="store_true")
    p_prep.add_argument("--prepare_max_lines", type=int, default=None)

    p_tr = sub.add_parser("train", help="在 memmap 上训练 TinyGPT")
    _apply_common(p_tr)
    p_tr.add_argument("--text_field", type=str, default="text")
    p_tr.add_argument("--block_size", type=int, default=256)
    p_tr.add_argument("--n_layer", type=int, default=8)
    p_tr.add_argument("--n_head", type=int, default=8)
    p_tr.add_argument("--n_embd", type=int, default=512)
    p_tr.add_argument("--dropout", type=float, default=0.1)
    p_tr.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    p_tr.add_argument(
        "--pos_encoding",
        type=str,
        choices=("wpe", "rope"),
        default="wpe",
        help="wpe=baseline 绝对位置嵌入；rope=实验 A1，Q/K 上 RoPE、无 wpe",
    )
    p_tr.add_argument("--rope", action="store_true", help="实验 A1：等同 --pos_encoding rope（勿与滑动窗口同开）")
    p_tr.add_argument("--rope_theta", type=float, default=10000.0)
    p_tr.add_argument(
        "--attn_window",
        type=int,
        default=0,
        help="实验 A2：滑动窗口宽度 W；0=全长。须 wpe，勿与 --rope 同开",
    )
    p_tr.add_argument(
        "--sliding_window",
        action="store_true",
        help="实验 A2：等同 --attn_window 128（block_size=256 时看最近一半上下文）",
    )
    p_tr.add_argument(
        "--talking_heads",
        action="store_true",
        help="实验 B1：Talking-Heads；建议在 A1 上使用：同时加 --rope",
    )
    p_tr.add_argument(
        "--attn_dropout",
        type=float,
        default=0.0,
        help="仅注意力子层 dropout；0=与 --dropout 相同。B2 典型 0.2",
    )
    p_tr.add_argument(
        "--b2",
        action="store_true",
        help="实验 B2：A1(RoPE) + attn_dropout=0.2，resid/MLP 仍为 --dropout",
    )
    p_tr.add_argument("--norm_type", type=str, choices=("layernorm", "rmsnorm"), default="layernorm")
    p_tr.add_argument("--ffn_type", type=str, choices=("gelu", "swiglu"), default="gelu")
    p_tr.add_argument("--swiglu", action="store_true", help="等同 --ffn_type swiglu")
    p_tr.add_argument("--rmsnorm", action="store_true", help="等同 --norm_type rmsnorm")
    p_tr.add_argument(
        "--backbone",
        type=str,
        default="",
        choices=("", "a1b1", "a1b1_modern", "v2"),
        help="a1b1=RoPE+B1；a1b1_modern/v2=RoPE+B1+SwiGLU+RMSNorm（推荐下一版骨干）",
    )
    p_tr.add_argument("--learning_rate", type=float, default=6e-4)
    p_tr.add_argument("--weight_decay", type=float, default=0.1)
    p_tr.add_argument("--beta1", type=float, default=0.9)
    p_tr.add_argument("--beta2", type=float, default=0.95)
    p_tr.add_argument("--grad_clip", type=float, default=1.0)
    p_tr.add_argument("--warmup_steps", type=int, default=200)
    p_tr.add_argument("--lr_decay_ratio", type=float, default=0.1)
    p_tr.add_argument("--max_steps", type=int, default=20_000)
    p_tr.add_argument("--batch_size", type=int, default=32)
    p_tr.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--device", type=str, default="cuda")
    p_tr.add_argument("--eval_interval", type=int, default=200)
    p_tr.add_argument("--eval_iters", type=int, default=200)
    p_tr.add_argument("--log_interval", type=int, default=10)
    p_tr.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p_tr.add_argument("--no_loss_plot", action="store_true", help="不生成 loss_curve.png（仍写 loss_history.json）")
    p_tr.add_argument(
        "--test_jsonl",
        type=str,
        default="",
        help="测试集 jsonl；默认 {data_root}/{split}/test.jsonl，不存在则回退 data/test.jsonl",
    )
    p_tr.add_argument("--no_test_after_train", action="store_true", help="训练结束后不在测试集上评测 best.pt")
    p_tr.add_argument("--resume", type=str, default="")

    args = parser.parse_args()
    cfg = _cfg_from_ns(args)
    if args.command == "prepare":
        prepare(cfg)
        return
    if args.command == "train":
        try:
            train(cfg)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        return
    raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
