#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按「文档长度」「前缀长度」「文档内位置」分桶，对比多个 checkpoint 的 CE/PPL。
用于汇报：说明 A2（滑动窗口）在何种样本上劣于 A1（RoPE）。

示例：
  python scripts/eval_prefix_buckets.py \\
    --jsonl /root/autodl-tmp/manifest_full_out/test.jsonl \\
    --ckpt_a1 checkpoints/20260520_114116_32576206/best.pt \\
    --ckpt_a2 checkpoints/20260520_143243_35029424/best.pt \\
    --label_a1 A1_rope --label_a2 A2_window \\
    --max_lines 8000 \\
    --output_dir ./badcase_runs/a1_vs_a2/prefix_buckets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.config import GPTConfig
from model.gpt import TinyGPT
from tokenizer import Tokenizer, load_tokenizer, vocab_size_of

_LN2 = math.log(2.0)

DOC_LEN_BUCKETS = (
    ("1-32", 1, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257+", 257, 10**9),
)

PREFIX_BUCKETS = (
    ("P<=16", 0, 16),
    ("17-32", 17, 32),
    ("33-48", 33, 48),
    ("49-96", 49, 96),
    ("97+", 97, 10**9),
)

POS_IN_DOC_BUCKETS = (
    ("pos 0-47", 0, 47),
    ("pos 48-127", 48, 127),
    ("pos 128+", 128, 10**9),
)

FIXED_PREFIX_LENS = (16, 32, 48, 64, 96)


@dataclass
class BucketAcc:
    sum_ce: float = 0.0
    n: int = 0

    def add(self, ce: float) -> None:
        self.sum_ce += ce
        self.n += 1

    def mean(self) -> float | None:
        return self.sum_ce / self.n if self.n else None


@dataclass
class EvalAccum:
    doc_len: dict[str, BucketAcc] = field(default_factory=lambda: {b[0]: BucketAcc() for b in DOC_LEN_BUCKETS})
    prefix_len: dict[str, BucketAcc] = field(default_factory=lambda: {b[0]: BucketAcc() for b in PREFIX_BUCKETS})
    pos_in_doc: dict[str, BucketAcc] = field(default_factory=lambda: {b[0]: BucketAcc() for b in POS_IN_DOC_BUCKETS})
    fixed_prefix_suffix: dict[int, BucketAcc] = field(
        default_factory=lambda: {p: BucketAcc() for p in FIXED_PREFIX_LENS}
    )
    n_docs: int = 0
    n_docs_skipped_long: int = 0


def _ppl(ce: float | None) -> float | None:
    if ce is None:
        return None
    return math.exp(ce) if ce < 80 else float("inf")


def _bucket_name(length: int, spec: tuple) -> str | None:
    for name, lo, hi in spec:
        if lo <= length <= hi:
            return name
    return None


def _load_model(ckpt_path: str, device: torch.device) -> tuple[TinyGPT, Tokenizer, dict[str, Any]]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt["gpt_config"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    tc = ckpt.get("train_config") or {}
    tok_dir = tc.get("tokenizer_dir") or ""
    if not tok_dir:
        dr = tc.get("data_root", "/root/autodl-tmp")
        backend = tc.get("tokenizer_backend", "char")
        sub = "gpt2_tokenizer" if backend == "gpt2" else "char_tokenizer"
        tok_dir = os.path.join(dr, sub)
    tok = load_tokenizer(tok_dir)
    meta = {
        "pos_encoding": cfg.pos_encoding,
        "attn_window": cfg.attn_window,
        "block_size": cfg.block_size,
    }
    return model, tok, meta


@torch.no_grad()
def _per_token_ce(model: TinyGPT, ids: list[int], device: torch.device, block_size: int) -> list[float]:
    """非重叠 block 切分，与 train.py 测试一致；返回每个被预测位置的 CE。"""
    ces: list[float] = []
    buf = ids
    while len(buf) >= 2:
        T = min(block_size, len(buf) - 1)
        if T < 1:
            break
        x = torch.tensor([buf[:T]], dtype=torch.long, device=device)
        y = torch.tensor([buf[1 : T + 1]], dtype=torch.long, device=device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
        ces.extend(ce.cpu().tolist())
        buf = buf[T:]
    return ces


def _eval_jsonl(
    model: TinyGPT,
    tok: Tokenizer,
    jsonl_path: str,
    device: torch.device,
    block_size: int,
    max_lines: int,
) -> EvalAccum:
    acc = EvalAccum()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if max_lines > 0 and acc.n_docs >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue
            ids = tok.encode(text, add_bos=True, add_eos=True)
            if len(ids) < 3:
                continue
            acc.n_docs += 1
            doc_len = len(ids)
            if doc_len > block_size * 4:
                acc.n_docs_skipped_long += 1

            ces = _per_token_ce(model, ids, device, block_size)
            if not ces:
                continue

            doc_bucket = _bucket_name(doc_len, DOC_LEN_BUCKETS)
            if doc_bucket:
                for c in ces:
                    acc.doc_len[doc_bucket].add(c)

            for i, c in enumerate(ces):
                pb = _bucket_name(i + 1, PREFIX_BUCKETS)
                if pb:
                    acc.prefix_len[pb].add(c)
                if doc_len >= 150:
                    pos_b = _bucket_name(i, POS_IN_DOC_BUCKETS)
                    if pos_b:
                        acc.pos_in_doc[pos_b].add(c)

            for P in FIXED_PREFIX_LENS:
                if len(ces) > P + 8:
                    for c in ces[P:]:
                        acc.fixed_prefix_suffix[P].add(c)

    return acc


def _serialize_buckets(buckets: dict[str, BucketAcc]) -> dict[str, Any]:
    out = {}
    for name, b in buckets.items():
        m = b.mean()
        out[name] = {
            "n_tokens": b.n,
            "mean_ce": m,
            "ppl": _ppl(m),
        }
    return out


def _report_one(label: str, meta: dict[str, Any], acc: EvalAccum) -> dict[str, Any]:
    return {
        "label": label,
        "model": meta,
        "n_docs": acc.n_docs,
        "n_docs_skipped_long_hint": acc.n_docs_skipped_long,
        "by_doc_token_len": _serialize_buckets(acc.doc_len),
        "by_prefix_token_index": _serialize_buckets(acc.prefix_len),
        "by_position_in_long_doc": _serialize_buckets(acc.pos_in_doc),
        "suffix_ce_after_fixed_prefix": {
            str(P): {
                "n_tokens": acc.fixed_prefix_suffix[P].n,
                "mean_ce": (m := acc.fixed_prefix_suffix[P].mean()),
                "ppl": _ppl(m),
            }
            for P in FIXED_PREFIX_LENS
        },
    }


def _plot_compare(
    reports: list[dict[str, Any]],
    key: str,
    title: str,
    out_path: str,
    ylabel: str = "mean CE (nat)",
) -> None:
    bucket_names = list(reports[0][key].keys())
    x = range(len(bucket_names))
    w = 0.8 / len(reports)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, rep in enumerate(reports):
        ys = [rep[key][b]["mean_ce"] for b in bucket_names]
        off = (i - (len(reports) - 1) / 2) * w
        ax.bar([xi + off for xi in x], ys, width=w, label=rep["label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(bucket_names, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="分桶对比 checkpoint 测试 CE/PPL")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt_a1", type=str, required=True)
    p.add_argument("--ckpt_a2", type=str, default="")
    p.add_argument("--ckpt_baseline", type=str, default="")
    p.add_argument("--label_a1", type=str, default="A1")
    p.add_argument("--label_a2", type=str, default="A2")
    p.add_argument("--label_baseline", type=str, default="baseline")
    p.add_argument("--max_lines", type=int, default=8000, help="0=全量（很慢）")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    jobs: list[tuple[str, str]] = [(args.label_a1, args.ckpt_a1)]
    if args.ckpt_a2:
        jobs.append((args.label_a2, args.ckpt_a2))
    if args.ckpt_baseline:
        jobs.append((args.label_baseline, args.ckpt_baseline))

    reports: list[dict[str, Any]] = []
    for label, ckpt in jobs:
        print(f"评测 {label}: {ckpt}")
        model, tok, meta = _load_model(ckpt, device)
        acc = _eval_jsonl(model, tok, args.jsonl, device, meta["block_size"], args.max_lines)
        rep = _report_one(label, meta, acc)
        reports.append(rep)
        print(f"  docs={acc.n_docs}")

    out = {
        "test_jsonl": os.path.abspath(args.jsonl),
        "max_lines": args.max_lines,
        "interpretation": {
            "by_doc_token_len": "文档越长，A2(W=128)越难 attend 文首 → 期望 A2 在 129+ 桶更差",
            "by_prefix_token_index": "位置索引越大=离开头越远；与 badcase 短前缀后续写相关",
            "suffix_ce_after_fixed_prefix": "固定前缀长 P，只统计 P 之后 token 的 CE（条件续写难度）",
            "by_position_in_long_doc": "仅 doc>=150 token；pos128+ 在 A2 下无法看全文开头",
        },
        "models": reports,
    }
    json_path = os.path.join(args.output_dir, "prefix_bucket_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if len(reports) >= 2:
        _plot_compare(
            reports,
            "by_doc_token_len",
            "Mean CE by document token length",
            os.path.join(args.output_dir, "chart_doc_len.png"),
        )
        _plot_compare(
            reports,
            "by_position_in_long_doc",
            "Mean CE by position (docs with len>=150)",
            os.path.join(args.output_dir, "chart_pos_in_long_doc.png"),
        )
        fixed = {
            rep["label"]: rep["suffix_ce_after_fixed_prefix"]
            for rep in reports
        }
        fig, ax = plt.subplots(figsize=(8, 4))
        Ps = [str(p) for p in FIXED_PREFIX_LENS]
        x = range(len(Ps))
        w = 0.8 / len(reports)
        for i, rep in enumerate(reports):
            ys = [rep["suffix_ce_after_fixed_prefix"][str(p)]["mean_ce"] for p in FIXED_PREFIX_LENS]
            off = (i - (len(reports) - 1) / 2) * w
            ax.bar([xi + off for xi in x], ys, width=w, label=rep["label"])
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"prefix={p}" for p in FIXED_PREFIX_LENS])
        ax.set_ylabel("suffix mean CE")
        ax.set_title("CE on tokens after fixed prefix length P")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "chart_fixed_prefix.png"), dpi=140)
        plt.close(fig)

    print(f"已写 {json_path}")


if __name__ == "__main__":
    main()
