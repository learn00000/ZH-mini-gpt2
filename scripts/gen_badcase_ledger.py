#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 checkpoint 按「体裁」采样若干条语料，用每行正文前缀做条件生成，输出 badcase 台账：
  - badcase_ledger.jsonl（完整字段）
  - badcase_ledger.csv（表格，可用 Excel / Numbers 打开）
  - badcase_ledger.html（浏览器里看长文本更清晰）
  - badcase_summary.json

示例（在仓库根目录执行，一行命令）：
  python scripts/gen_badcase_ledger.py --ckpt checkpoints/.../best.pt \\
    --jsonl /root/autodl-tmp/data/test.jsonl --tokenizer_dir /root/autodl-tmp/char_tokenizer \\
    --output_dir ./badcase_runs/run1

若 JSON 无 type 字段，样本会落在 other；可用 --rows_other 多抽几条。溯源依赖 JSON 内
是否自带 id/url/source 等字段；若无则仅能定位到「文件路径 + 行号」。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.config import GPTConfig
from model.gpt import TinyGPT
from tokenizer import Tokenizer, load_tokenizer, vocab_size_of


CANON_TYPES = ("news", "wiki", "comment", "webtext", "other")

_TYPE_ALIASES: Dict[str, str] = {
    "news": "news",
    "新闻": "news",
    "wiki": "wiki",
    "维基": "wiki",
    "百科": "wiki",
    "comment": "comment",
    "评论": "comment",
    "社交": "comment",
    "social": "comment",
    "webtext": "webtext",
    "web": "webtext",
    "网页": "webtext",
}

# 从原始 JSON 里尽量抄出「数据来源」线索（有则写进台账，无则只有 文件:行号）
_PROVENANCE_KEYS: Tuple[str, ...] = (
    "id",
    "_id",
    "url",
    "link",
    "source",
    "source_file",
    "source_name",
    "doc_id",
    "dataset",
    "title",
    "from",
    "path",
    "origin",
    "file",
    "domain",
)


def normalize_doc_type(raw: Any) -> str:
    if raw is None:
        return "other"
    s = str(raw).strip().lower()
    if s in _TYPE_ALIASES:
        return _TYPE_ALIASES[s]
    t = str(raw).strip()
    if t in _TYPE_ALIASES:
        return _TYPE_ALIASES[t]
    if s in CANON_TYPES:
        return s
    return "other"


def extract_provenance(
    obj: Dict[str, Any],
    *,
    skip_keys: frozenset[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in _PROVENANCE_KEYS:
        if k in skip_keys or k not in obj:
            continue
        v = obj[k]
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            s = json.dumps(v, ensure_ascii=False)
            if len(s) <= 400:
                out[k] = v
            else:
                out[k] = s[:400] + "…"
    return out


def build_source_ref(jsonl_path: str, line_no: int) -> str:
    """人类可读、可 grep 的定位串。"""
    return f"{os.path.abspath(jsonl_path)}:{line_no}"


def _max_char_run(s: str) -> int:
    if not s:
        return 0
    best = cur = 1
    prev = s[0]
    for ch in s[1:]:
        if ch == prev:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
            prev = ch
    return best


def _has_repeated_span(s: str, span: int = 24, min_occurrences: int = 2) -> bool:
    if len(s) < span * min_occurrences:
        return False
    seen: set[str] = set()
    dup: set[str] = set()
    for i in range(0, len(s) - span + 1):
        chunk = s[i : i + span]
        if chunk in seen:
            dup.add(chunk)
            if len(dup) >= 1:
                return True
        seen.add(chunk)
    return False


def _punctuation_spam(s: str) -> bool:
    return bool(re.search(r"[。！？，、；：,.!?;:]{8,}", s))


def _weird_whitespace_ratio(s: str) -> float:
    if not s:
        return 0.0
    ws = sum(1 for c in s if c.isspace())
    return ws / len(s)


def heuristic_flags(generated_only: str) -> Dict[str, Any]:
    g = generated_only
    flags: Dict[str, Any] = {}
    run = _max_char_run(g)
    flags["max_same_char_run"] = run
    flags["heavy_char_repeat"] = run >= 10

    flags["repeated_span_24"] = _has_repeated_span(g, span=24, min_occurrences=2)
    flags["repeated_span_12"] = _has_repeated_span(g, span=12, min_occurrences=3)

    flags["punctuation_spam"] = _punctuation_spam(g)
    flags["weird_whitespace_ratio"] = round(_weird_whitespace_ratio(g), 4)
    flags["weird_whitespace"] = flags["weird_whitespace_ratio"] > 0.25

    u = len(set(g)) / max(len(g), 1)
    flags["unique_char_ratio"] = round(u, 4)
    flags["low_diversity"] = len(g) >= 80 and u < 0.12

    cjk = sum(1 for c in g if "\u4e00" <= c <= "\u9fff")
    flags["cjk_ratio"] = round(cjk / max(len(g), 1), 4)
    flags["little_chinese"] = len(g) >= 40 and flags["cjk_ratio"] < 0.15

    flags["any_issue_hint"] = bool(
        flags["heavy_char_repeat"]
        or flags["repeated_span_24"]
        or flags["punctuation_spam"]
        or flags["weird_whitespace"]
        or flags["low_diversity"]
        or flags["little_chinese"]
    )
    return flags


def flags_to_short(flags: Dict[str, Any]) -> str:
    parts: List[str] = []
    if flags.get("heavy_char_repeat"):
        parts.append("char_rep")
    if flags.get("repeated_span_24"):
        parts.append("rep24")
    if flags.get("repeated_span_12"):
        parts.append("rep12")
    if flags.get("punctuation_spam"):
        parts.append("punct")
    if flags.get("weird_whitespace"):
        parts.append("ws")
    if flags.get("low_diversity"):
        parts.append("lowdiv")
    if flags.get("little_chinese"):
        parts.append("nonzh")
    if not parts:
        parts.append("ok")
    return ";".join(parts)


def load_jsonl_grouped(
    path: str,
    text_field: str,
    type_field: str,
    max_lines: Optional[int],
) -> Tuple[Dict[str, List[Tuple[int, Dict[str, Any]]]], Counter[str]]:
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    raw_counter: Counter[str] = Counter()

    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if max_lines is not None and n >= max_lines:
                break
            line = raw.strip()
            if not line:
                continue
            n += 1
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            text = obj.get(text_field, "")
            if not isinstance(text, str) or not text.strip():
                continue
            raw_type = obj.get(type_field)
            norm = normalize_doc_type(raw_type)
            raw_counter[str(raw_type)] += 1
            grouped[norm].append((line_no, obj))

    return dict(grouped), raw_counter


def pick_samples(
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    rows_per_type: int,
    rows_other: int,
    seed: int,
) -> List[Tuple[str, int, Dict[str, Any]]]:
    rng = random.Random(seed)
    out: List[Tuple[str, int, Dict[str, Any]]] = []
    for t in ("news", "wiki", "comment", "webtext"):
        rows = grouped.get(t, [])
        if not rows:
            continue
        k = min(rows_per_type, len(rows))
        chosen = rng.sample(rows, k=k) if len(rows) > k else list(rows)
        for line_no, obj in chosen:
            out.append((t, line_no, obj))

    other_rows = grouped.get("other", [])
    if other_rows:
        k = min(rows_other, len(other_rows))
        chosen = rng.sample(other_rows, k=k) if len(other_rows) > k else list(other_rows)
        for line_no, obj in chosen:
            out.append(("other", line_no, obj))

    return out


def build_idx_from_prefix(
    tok: Tokenizer,
    text: str,
    prefix_chars: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    if prefix_chars <= 0:
        return torch.tensor([[tok.bos_id]], dtype=torch.long, device=device)
    prefix = text[:prefix_chars]
    ids = tok.encode(prefix, add_bos=True, add_eos=False)
    if len(ids) > block_size:
        ids = ids[-block_size:]
    return torch.tensor([ids], dtype=torch.long, device=device)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "ledger_id",
        "sample_type",
        "raw_type_field",
        "source_ref",
        "source_jsonl",
        "source_line",
        "provenance_json",
        "flags_short",
        "any_issue_hint",
        "gen_chars",
        "input_prefix_text",
        "generated_text",
        "human_note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            prov = r.get("source_provenance") or {}
            w.writerow(
                {
                    "ledger_id": r["ledger_id"],
                    "sample_type": r["sample_type"],
                    "raw_type_field": r.get("raw_type_field"),
                    "source_ref": r.get("source_ref"),
                    "source_jsonl": r.get("source_jsonl"),
                    "source_line": r.get("source_line"),
                    "provenance_json": json.dumps(prov, ensure_ascii=False) if prov else "",
                    "flags_short": r.get("flags_short"),
                    "any_issue_hint": r.get("heuristic_flags", {}).get("any_issue_hint"),
                    "gen_chars": len(r.get("generated_text") or ""),
                    "input_prefix_text": r.get("input_prefix_text"),
                    "generated_text": r.get("generated_text"),
                    "human_note": r.get("human_note", ""),
                }
            )


def write_html(path: str, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>badcase ledger</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;}",
        "table{border-collapse:collapse;width:100%;background:#fff;}",
        "th,td{border:1px solid #ccc;padding:8px;vertical-align:top;}",
        "th{background:#eee;position:sticky;top:0;}",
        ".mono{font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap;word-break:break-all;}",
        ".narrow{max-width:120px;}",
        ".hint{color:#b45309;}</style></head><body>",
        f"<h2>badcase 台账</h2><p class='mono'>{html.escape(json.dumps({k: meta[k] for k in ('ckpt','jsonl','created_at') if k in meta}, ensure_ascii=False))}</p>",
        "<table><thead><tr>",
        "<th>id</th><th>体裁</th><th>原始type</th><th>定位</th><th>溯源字段</th><th>flags</th><th>前缀</th><th>续写</th></tr></thead><tbody>",
    ]
    for r in rows:
        prov = r.get("source_provenance") or {}
        prov_s = html.escape(json.dumps(prov, ensure_ascii=False)) if prov else "（无额外字段，仅文件+行号）"
        flags = r.get("heuristic_flags") or {}
        hint = " class='hint'" if flags.get("any_issue_hint") else ""
        parts.append("<tr>")
        parts.append(f"<td class='narrow'>{r['ledger_id']}</td>")
        parts.append(f"<td>{html.escape(str(r['sample_type']))}</td>")
        parts.append(f"<td class='mono'>{html.escape(str(r.get('raw_type_field')))}</td>")
        parts.append(f"<td class='mono'>{html.escape(str(r.get('source_ref')))}</td>")
        parts.append(f"<td class='mono'>{prov_s}</td>")
        parts.append(f"<td{hint} class='mono'>{html.escape(r.get('flags_short',''))}</td>")
        parts.append(f"<td class='mono'>{html.escape(r.get('input_prefix_text') or '')}</td>")
        parts.append(f"<td class='mono'>{html.escape(r.get('generated_text') or '')}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按体裁采样生成，写 badcase 台账（JSONL + CSV + HTML）")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--tokenizer_dir", type=str, default="")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--type_field", type=str, default="type")
    p.add_argument(
        "--rows_per_type",
        type=int,
        default=12,
        help="news/wiki/comment/webtext 每类最多采样条数",
    )
    p.add_argument(
        "--rows_other",
        type=int,
        default=40,
        help="other 类（含无 type 字段）最多采样条数，通常应更大",
    )
    p.add_argument("--prefix_chars", type=int, default=48)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_read_lines", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_s = args.device
    if device_s.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU。")
        device = torch.device("cpu")
    else:
        device = torch.device(device_s)

    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)

    gpt_dict = ckpt.get("gpt_config") or {}
    cfg = GPTConfig(**gpt_dict)
    model = TinyGPT(cfg).to(device)
    _load_state(model, ckpt)

    tc = ckpt.get("train_config") or {}
    tok_dir = args.tokenizer_dir.strip()
    if not tok_dir:
        tok_dir = (tc.get("tokenizer_dir") or "").strip()
    if not tok_dir:
        dr = tc.get("data_root", "/root/autodl-tmp")
        backend = tc.get("tokenizer_backend", "char")
        sub = "gpt2_tokenizer" if backend == "gpt2" else "char_tokenizer"
        tok_dir = os.path.join(dr, sub)
    if not tok_dir or not os.path.isdir(tok_dir):
        raise SystemExit(f"tokenizer 目录无效: {tok_dir!r}，请传 --tokenizer_dir")

    tok = load_tokenizer(tok_dir)

    out_dir = args.output_dir.strip()
    if not out_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.getcwd(), "badcase_runs", ts)
    os.makedirs(out_dir, exist_ok=True)

    jsonl_abs = os.path.abspath(args.jsonl)
    grouped, raw_type_hist = load_jsonl_grouped(
        args.jsonl,
        text_field=args.text_field,
        type_field=args.type_field,
        max_lines=args.max_read_lines,
    )

    counts_by_norm = {k: len(v) for k, v in grouped.items()}
    print("=== JSONL 内可分组条数（按规范体裁）===")
    for k in CANON_TYPES:
        print(f"  {k}: {counts_by_norm.get(k, 0)}")
    print(f"  （{args.type_field} 原始值分布见 summary）")

    samples = pick_samples(grouped, args.rows_per_type, args.rows_other, args.seed)
    if not samples:
        raise SystemExit("没有可用样本：请检查 JSONL 是否含非空 text。")

    skip_prov = frozenset({args.text_field, args.type_field})
    ledger_path = os.path.join(out_dir, "badcase_ledger.jsonl")
    csv_path = os.path.join(out_dir, "badcase_ledger.csv")
    html_path = os.path.join(out_dir, "badcase_ledger.html")
    summary_path = os.path.join(out_dir, "badcase_summary.json")

    meta: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ckpt": os.path.abspath(args.ckpt),
        "jsonl": jsonl_abs,
        "tokenizer_dir": os.path.abspath(tok_dir),
        "text_field": args.text_field,
        "type_field": args.type_field,
        "rows_per_type": args.rows_per_type,
        "rows_other": args.rows_other,
        "prefix_chars": args.prefix_chars,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "gpt_config": asdict(cfg),
        "counts_by_norm_type_in_file": counts_by_norm,
        "raw_type_field_counts": dict(raw_type_hist.most_common()),
        "source_provenance_note": (
            "台账中的 source_ref = 绝对路径:行号；若 JSON 行内包含 id/url/source 等字段，"
            "会写入 source_provenance。若均为空，则无法比文件行号更细，需在造数据时写入。"
        ),
    }

    ledger_rows: List[Dict[str, Any]] = []
    flag_counter: Counter[str] = Counter()
    model.eval()

    for ledger_id, (norm_type, line_no, obj) in enumerate(samples, start=1):
        text = obj.get(args.text_field, "")
        assert isinstance(text, str)

        prefix = "" if args.prefix_chars <= 0 else text[: args.prefix_chars]
        idx = build_idx_from_prefix(tok, text, args.prefix_chars, cfg.block_size, device)
        in_len = int(idx.shape[1])

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=max(args.temperature, 1e-8),
                top_k=args.top_k if args.top_k > 0 else None,
            )

        full_ids = out[0].tolist()
        gen_ids = full_ids[in_len:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        flags = heuristic_flags(gen_text)
        if flags.get("any_issue_hint"):
            flag_counter["rows_with_any_hint"] += 1
        for fk, fv in flags.items():
            if fk.endswith("_hint") or fk.startswith("max_") or fk.endswith("_ratio"):
                continue
            if isinstance(fv, bool) and fv:
                flag_counter[fk] += 1

        provenance = extract_provenance(obj, skip_keys=skip_prov)
        source_ref = build_source_ref(args.jsonl, line_no)

        row = {
            "ledger_id": ledger_id,
            "sample_type": norm_type,
            "raw_type_field": obj.get(args.type_field),
            "source_jsonl": jsonl_abs,
            "source_line": line_no,
            "source_ref": source_ref,
            "source_provenance": provenance,
            "prefix_chars_setting": args.prefix_chars,
            "input_prefix_text": prefix,
            "condition_token_len": in_len,
            "generated_text": gen_text,
            "heuristic_flags": flags,
            "flags_short": flags_to_short(flags),
            "human_note": "",
        }
        ledger_rows.append(row)

    with open(ledger_path, "w", encoding="utf-8") as lf:
        for row in ledger_rows:
            lf.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(csv_path, ledger_rows)
    write_html(html_path, ledger_rows, meta)

    meta["ledger_path"] = os.path.abspath(ledger_path)
    meta["csv_path"] = os.path.abspath(csv_path)
    meta["html_path"] = os.path.abspath(html_path)
    meta["summary_path"] = os.path.abspath(summary_path)
    meta["num_ledger_rows"] = len(ledger_rows)
    meta["flag_counter"] = dict(flag_counter)
    meta["heuristic_legend"] = {
        "heavy_char_repeat": "同一字符连续出现过多",
        "repeated_span_24": "较长子串重复出现",
        "repeated_span_12": "12 字子串多次重复",
        "punctuation_spam": "连续标点过长",
        "weird_whitespace": "空白比例异常",
        "low_diversity": "字符种类占比过低（长文本）",
        "little_chinese": "续写里中文占比过低",
        "any_issue_hint": "任一上述强提示为真",
    }

    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(meta, sf, ensure_ascii=False, indent=2)

    print(f"\n已写 JSONL: {ledger_path}")
    print(f"已写 CSV（表格）: {csv_path}")
    print(f"已写 HTML（推荐阅读）: {html_path}")
    print(f"已写汇总: {summary_path}")
    print(f"共 {len(ledger_rows)} 条；带 any_issue_hint: {flag_counter.get('rows_with_any_hint', 0)}")


def _load_state(model: TinyGPT, ckpt: Dict[str, Any]) -> None:
    if "model" not in ckpt:
        raise SystemExit("checkpoint 中缺少 model state_dict")
    model.load_state_dict(ckpt["model"], strict=True)


if __name__ == "__main__":
    main()
