#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 checkpoint 抽取因果注意力权重，生成图表以验证 badcase 相关假设。

假设 ↔ 图表：
  A 跑题/硬接  → prefix_attention_ratio（生成段是否仍看前缀）
                 + 远距离注意力质量（非前缀区域的平均权重）
  B 机械重复    → 注意力熵（过低=分布过尖）+ 重复字处对「相同历史字」的注意力峰

用法（仓库根目录）：
  python scripts/visualize_attention.py \\
    --ckpt checkpoints/20260519_195338_52068640/best.pt \\
    --prefix "甘肃省敦煌种业股份有限公司(600354)2014年年度股东大会" \\
    --generated_text "（续写片段，可选）" \\
    --output_dir ./attn_viz/demo1

  # 从 badcase 台账取一条
  python scripts/visualize_attention.py \\
    --ckpt checkpoints/20260519_195338_52068640/best.pt \\
    --ledger ./badcase_runs/run3_newdata_newmodel/badcase_ledger.jsonl \\
    --ledger_id 8 \\
    --layer -1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 与 scripts/plot_report_figures.py 一致：显式注册 .ttc，否则热力图轴标签中文会显示为方框
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def _setup_cjk_font() -> None:
    if os.path.isfile(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        name = fm.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_setup_cjk_font()
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.config import GPTConfig
from model.gpt import TinyGPT
from tokenizer import Tokenizer, load_tokenizer


def _load_model(ckpt_path: str, device: torch.device) -> tuple[TinyGPT, GPTConfig, Tokenizer]:
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
        tok_dir = os.path.join(dr, "char_tokenizer")
    tok = load_tokenizer(tok_dir)
    return model, cfg, tok


def _encode_sequence(tok: Tokenizer, prefix: str, generated: str) -> tuple[list[int], int]:
    ids = tok.encode(prefix, add_bos=True, add_eos=False)
    prefix_len = len(ids)
    if generated:
        ids.extend(tok.encode(generated, add_bos=False, add_eos=False))
    return ids, prefix_len


def _token_labels(tok: Tokenizer, ids: list[int], max_labels: int = 80) -> list[str]:
    labels = []
    for i in ids:
        t = tok.id_to_token.get(i, "?")
        if t in ("<pad>", "<bos>", "<eos>", "<unk>"):
            labels.append(t)
        else:
            labels.append(t if len(t) <= 8 else t[:8])
    if len(labels) > max_labels:
        step = max(1, len(labels) // max_labels)
        labels = [labels[i] if i % step == 0 else "" for i in range(len(labels))]
    return labels


def _avg_heads(att: torch.Tensor) -> np.ndarray:
    """(B, nh, T, T) -> (T, T) numpy，对 batch 与 head 平均。"""
    return att[0].mean(dim=0).cpu().numpy()


def _entropy_per_row(att: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """att: (T, T) 因果下每行是 query 对 key 的分布。"""
    p = np.clip(att, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def _prefix_ratio_per_row(att: np.ndarray, prefix_len: int) -> np.ndarray:
    """每个 query 位置对 key∈[0, prefix_len) 的注意力质量和。"""
    T = att.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for q in range(T):
        out[q] = att[q, : min(prefix_len, q + 1)].sum()
    return out


def _far_mass_per_row(att: np.ndarray, prefix_len: int) -> np.ndarray:
    """每个 query 对 key>=prefix_len 且 <=q 的注意力质量和（生成段内「非前缀」历史）。"""
    T = att.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for q in range(T):
        if q < prefix_len:
            out[q] = 0.0
        else:
            out[q] = att[q, prefix_len : q + 1].sum()
    return out


def _repeat_peak_score(ids: list[int], att: np.ndarray, gen_start: int) -> list[dict[str, Any]]:
    """
    在生成段：若当前字与更早生成字相同，看当前行对「最早相同字位置」的注意力是否偏高。
  返回若干可疑点供报告。
    """
    hits: list[dict[str, Any]] = []
    for q in range(gen_start, len(ids)):
        ch_id = ids[q]
        if ch_id < 4:
            continue
        for k in range(gen_start, q):
            if ids[k] == ch_id:
                attn_to_k = float(att[q, k])
                row_sum = float(att[q, : q + 1].sum()) or 1.0
                hits.append(
                    {
                        "query_pos": q,
                        "key_pos": k,
                        "char_id": ids[q],
                        "attn_to_repeat": attn_to_k,
                        "row_mass_on_repeat": attn_to_k / row_sum,
                    }
                )
                break
    hits.sort(key=lambda x: x["attn_to_repeat"], reverse=True)
    return hits[:15]


def plot_heatmap(
    att: np.ndarray,
    ids: list[int],
    tok: Tokenizer,
    prefix_len: int,
    out_path: str,
    title: str,
) -> None:
    T = att.shape[0]
    fig, ax = plt.subplots(figsize=(min(14, T * 0.15 + 4), min(12, T * 0.12 + 3)))
    im = ax.imshow(att, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.01, att.max()))
    ax.set_title(title)
    ax.set_xlabel("key position (token index)")
    ax.set_ylabel("query position")
    if prefix_len > 0 and prefix_len < T:
        ax.axhline(prefix_len - 0.5, color="red", linestyle="--", linewidth=0.8, label="prefix|gen")
        ax.axvline(prefix_len - 0.5, color="red", linestyle="--", linewidth=0.8)
    labels = _token_labels(tok, ids)
    step = max(1, T // 40)
    ticks = list(range(0, T, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_curves(
    steps: np.ndarray,
    series: Dict[str, np.ndarray],
    out_path: str,
    title: str,
    ylabel: str,
    vline: Optional[int] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, y in series.items():
        ax.plot(steps, y, label=name, linewidth=1.2)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", label="prefix|gen")
    ax.set_title(title)
    ax.set_xlabel("token position")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _decorate_repeat_hits(tok: Tokenizer, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for h in hits:
        cid = h["char_id"]
        h2 = dict(h)
        h2["char"] = tok.id_to_token.get(cid, "?")
        out.append(h2)
    return out


def load_ledger_row(path: str, ledger_id: int) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if int(row["ledger_id"]) == ledger_id:
                return row
    raise SystemExit(f"ledger_id={ledger_id} 不在 {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="注意力可视化（验证 badcase 假设）")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tokenizer_dir", type=str, default="")
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--generated_text", type=str, default="")
    p.add_argument("--ledger", type=str, default="", help="badcase_ledger.jsonl")
    p.add_argument("--ledger_id", type=int, default=0)
    p.add_argument("--batch_all", action="store_true", help="对 ledger 中每条各出一组图")
    p.add_argument("--layer", type=int, default=-1, help="画哪一层，默认最后一层")
    p.add_argument("--max_len", type=int, default=128, help="序列过长时截断尾部")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _run_one(
    model: TinyGPT,
    cfg: GPTConfig,
    tok: Tokenizer,
    device: torch.device,
    prefix: str,
    generated: str,
    layer_arg: int,
    max_len: int,
    output_dir: str,
    meta_extra: dict[str, Any],
    ckpt_path: str,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    ids, prefix_len = _encode_sequence(tok, prefix, generated)
    if len(ids) > max_len:
        ids = ids[-max_len:]
        prefix_len = max(0, prefix_len - (len(ids) - max_len))
    if len(ids) < 2:
        raise ValueError("序列太短")
    if len(ids) > cfg.block_size:
        raise ValueError(f"序列长度 {len(ids)} > block_size {cfg.block_size}")

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, attns = model.forward_collect_attentions(idx)
    layer_idx = layer_arg if layer_arg >= 0 else len(attns) + layer_arg
    att = _avg_heads(attns[layer_idx])
    T = att.shape[0]
    gen_start = min(prefix_len, T)

    ent = _entropy_per_row(att)
    pref_ratio = _prefix_ratio_per_row(att, prefix_len)
    far_mass = _far_mass_per_row(att, prefix_len)
    repeat_hits = _decorate_repeat_hits(tok, _repeat_peak_score(ids, att, gen_start))

    pos = np.arange(T)
    gen_mask = pos >= prefix_len
    report = {
        "ckpt": os.path.abspath(ckpt_path),
        "layer": layer_idx,
        "n_layer": len(attns),
        "seq_len": T,
        "prefix_len": prefix_len,
        "meta": meta_extra,
        "metrics": {
            "entropy_mean_all": float(ent.mean()),
            "entropy_mean_gen": float(ent[gen_mask].mean()) if gen_mask.any() else None,
            "prefix_ratio_mean_gen": float(pref_ratio[gen_mask].mean()) if gen_mask.any() else None,
            "far_mass_mean_gen": float(far_mass[gen_mask].mean()) if gen_mask.any() else None,
        },
        "interpretation_guide": {
            "prefix_ratio_low_in_gen": "生成段很少看前缀 → 支持「跑题/硬接」假设 A",
            "entropy_low_in_gen": "生成段注意力熵过低 → 支持「机械重复/塌陷」假设 B",
            "repeat_hits_high": "生成字重复 attend 早期相同字 → 支持重复机制 B",
        },
        "top_repeat_peaks": repeat_hits,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    plot_heatmap(
        att,
        ids,
        tok,
        prefix_len,
        os.path.join(output_dir, f"heatmap_layer{layer_idx}.png"),
        f"Attention avg heads — layer {layer_idx}",
    )
    plot_curves(
        pos,
        {
            "entropy": ent,
            "prefix_mass": pref_ratio,
            "far_mass(non-prefix history)": far_mass,
        },
        os.path.join(output_dir, "curves_diagnosis.png"),
        "Diagnosis curves (see metrics.json)",
        "value",
        prefix_len,
    )

    layer_ent = []
    for a in attns:
        ae = _avg_heads(a)
        e = _entropy_per_row(ae)
        layer_ent.append(float(e[gen_mask].mean()) if gen_mask.any() else float(e.mean()))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(layer_ent)), layer_ent)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean entropy (gen region)")
    ax.set_title("Per-layer attention entropy (generation region)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "entropy_by_layer.png"), dpi=140)
    plt.close(fig)

    return report


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    model, cfg, tok = _load_model(args.ckpt, device)
    if args.tokenizer_dir:
        tok = load_tokenizer(args.tokenizer_dir)

    if args.ledger and args.batch_all:
        summaries: list[dict[str, Any]] = []
        with open(args.ledger, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            lid = int(row["ledger_id"])
            sub = os.path.join(args.output_dir, f"ledger_{lid}")
            meta = {
                "ledger_id": lid,
                "sample_type": row.get("sample_type"),
                "flags_short": row.get("flags_short"),
            }
            try:
                rep = _run_one(
                    model,
                    cfg,
                    tok,
                    device,
                    row.get("input_prefix_text", ""),
                    row.get("generated_text", ""),
                    args.layer,
                    args.max_len,
                    sub,
                    meta,
                    args.ckpt,
                )
                summaries.append(
                    {
                        "ledger_id": lid,
                        "flags_short": row.get("flags_short"),
                        "sample_type": row.get("sample_type"),
                        **rep["metrics"],
                    }
                )
                print(f"ok ledger {lid}")
            except Exception as e:
                summaries.append({"ledger_id": lid, "error": str(e)})
                print(f"skip ledger {lid}: {e}")
        with open(os.path.join(args.output_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"批量完成: {args.output_dir}/batch_summary.json")
        return

    prefix = args.prefix
    generated = args.generated_text
    meta_extra: dict[str, Any] = {}
    if args.ledger:
        row = load_ledger_row(args.ledger, args.ledger_id)
        prefix = row.get("input_prefix_text", "") or prefix
        generated = row.get("generated_text", "") or generated
        meta_extra = {
            "ledger_id": row.get("ledger_id"),
            "sample_type": row.get("sample_type"),
            "source_ref": row.get("source_ref"),
            "flags_short": row.get("flags_short"),
        }

    if not prefix:
        raise SystemExit("请提供 --prefix 或 --ledger + --ledger_id（或 --batch_all）")

    out_sub = args.output_dir
    if args.ledger and args.ledger_id:
        out_sub = os.path.join(args.output_dir, f"ledger_{args.ledger_id}")

    report = _run_one(
        model,
        cfg,
        tok,
        device,
        prefix,
        generated,
        args.layer,
        args.max_len,
        out_sub,
        meta_extra,
        args.ckpt,
    )
    print(f"已写: {out_sub}")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    if report.get("top_repeat_peaks"):
        print("top repeat peak:", report["top_repeat_peaks"][0])


if __name__ == "__main__":
    main()
