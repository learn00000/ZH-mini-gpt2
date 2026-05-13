#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLUECorpus2020-small 流式清洗脚本：输出 JSONL，支持按字节/条数试水采样。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import random
import re
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from tqdm import tqdm

# -----------------------------------------------------------------------------
# 正则与常量
# -----------------------------------------------------------------------------

# 常见 URL（含 http(s) 与裸 www.）
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\u201c\u201d\"'{}|\\^`\[\]\u300a\u300b]+",
    re.IGNORECASE,
)

# HTML 标签（简单非贪婪匹配，语料场景足够）
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# CJK 统一汉字
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

# 明显不可见 / 控制类：Cc + 常见零宽与双向标记
_ZERO_WIDTH_AND_BIDI = set(
    "\u200b\u200c\u200d\u200e\u200f"
    "\u202a\u202b\u202c\u202d\u202e"
    "\ufeff"
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


# -----------------------------------------------------------------------------
# 输入遍历
# -----------------------------------------------------------------------------


def iter_input_files(input_path: str, input_format: str) -> Iterator[Path]:
    """
    根据 input_format 递归收集 .txt 或 .jsonl 文件，路径排序以保证可复现。
    """
    root = Path(input_path).resolve()
    if root.is_file():
        suf = root.suffix.lower()
        if input_format == "txt" and suf == ".txt":
            yield root
        elif input_format == "jsonl" and suf == ".jsonl":
            yield root
        else:
            logging.warning("单文件扩展名与 --input_format=%s 不匹配: %s", input_format, root)
        return

    if not root.is_dir():
        logging.warning("输入路径不存在或不是目录/文件: %s", input_path)
        return

    # glob 不进入符号链接子目录；os.walk(followlinks=True) 可遍历 corpus_root 等链接布局
    suffix = ".txt" if input_format == "txt" else ".jsonl"
    collected: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        base = Path(dirpath)
        for name in filenames:
            if not name.lower().endswith(suffix):
                continue
            p = base / name
            try:
                if p.is_file():
                    collected.append(p)
            except OSError:
                continue
    for p in sorted(collected):
        yield p


def iter_lines(
    filepath: Path,
    input_format: str,
    text_key: str,
) -> Iterator[Tuple[str, int, str]]:
    """
    流式逐条产出 (文件路径字符串, 行号从1计, 原始文本片段)。
    txt: 每行一条；空行跳过。
    jsonl: 每行 JSON，取 text_key；解析失败则 warning 并跳过。
    """
    path_str = str(filepath)
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            if input_format == "txt":
                for lineno, line in enumerate(f, start=1):
                    if line.endswith("\n"):
                        line = line[:-1]
                    if line.endswith("\r"):
                        line = line[:-1]
                    if not line.strip():
                        continue
                    yield path_str, lineno, line
            else:
                for lineno, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.warning(
                            "JSON 解析失败 %s:%s — %s", path_str, lineno, e
                        )
                        continue
                    if not isinstance(obj, dict):
                        logging.warning(
                            "JSONL 行非对象 %s:%s，已跳过", path_str, lineno
                        )
                        continue
                    if text_key not in obj:
                        logging.warning(
                            "缺少字段 %r %s:%s，已跳过", text_key, path_str, lineno
                        )
                        continue
                    raw = obj[text_key]
                    if raw is None:
                        continue
                    if not isinstance(raw, str):
                        raw = str(raw)
                    yield path_str, lineno, raw
    except OSError as e:
        logging.warning("无法读取文件 %s: %s", path_str, e)


# -----------------------------------------------------------------------------
# 清洗与特征
# -----------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    去首尾空白、HTML 标签与实体、URL，压缩空白，去掉明显控制/不可见字符；
    保留中文、英文、数字及常见标点（不过度白名单过滤）。
    JSONL 单行输出：内部换行归一为空白。
    """
    if not text:
        return ""

    s = text.strip()
    if not s:
        return ""

    s = html.unescape(s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = _URL_RE.sub(" ", s)

    # 换行、制表等先统一成空格，再压缩
    s = re.sub(r"[\r\n\t\v\f]+", " ", s)

    out_chars: List[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat == "Cc":
            continue
        if ch in _ZERO_WIDTH_AND_BIDI:
            continue
        # 过滤其他格式控制字符（如部分 Cf），保留正常可见与标点
        if cat == "Cf":
            continue
        out_chars.append(ch)

    s = "".join(out_chars)
    s = re.sub(r" +", " ", s).strip()
    return s


def compute_zh_ratio(text: str) -> float:
    """中文字符数 / 总字符数；总长为 0 时返回 0.0。"""
    n = len(text)
    if n == 0:
        return 0.0
    zh = len(_ZH_CHAR_RE.findall(text))
    return zh / n


def make_signature(dedup_mode: str, text: str, prefix_len: int) -> str:
    """生成去重签名：md5 全文，或对前缀做 md5。"""
    if dedup_mode == "md5":
        payload = text.encode("utf-8")
    elif dedup_mode == "prefix":
        pl = max(1, prefix_len)
        payload = text[:pl].encode("utf-8")
    else:
        raise ValueError(f"未知 dedup_mode: {dedup_mode}")
    return hashlib.md5(payload).hexdigest()


def should_keep_text(
    cleaned: str,
    min_length: int,
    max_length: int,
    min_zh_ratio: float,
) -> Tuple[bool, Optional[str]]:
    """
    长度与中文占比检查（假定已非空）。
    返回 (是否保留, 若不保留则原因)。
    """
    L = len(cleaned)
    if L < min_length:
        return False, "too_short"
    if L > max_length:
        return False, "too_long"
    if compute_zh_ratio(cleaned) < min_zh_ratio:
        return False, "low_zh_ratio"
    return True, None


# -----------------------------------------------------------------------------
# Badcase 蓄水池抽样
# -----------------------------------------------------------------------------


def reservoir_offer_badcase(
    reservoir: List[Dict[str, str]],
    item: Dict[str, str],
    max_size: int,
    rng: random.Random,
    seen_count: int,
) -> None:
    """
    标准蓄水池抽样：seen_count 为已遇到的 badcase 总次数（1-based）。
    reservoir 长度不超过 max_size。
    """
    if max_size <= 0:
        return
    if seen_count <= max_size:
        reservoir.append(item)
        return
    j = rng.randint(0, seen_count - 1)
    if j < max_size:
        reservoir[j] = item


def write_badcase(path: str, items: List[Dict[str, str]]) -> None:
    """将 badcase 列表写入 JSONL。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not items:
        with open(path, "w", encoding="utf-8"):
            pass
        return
    with open(path, "w", encoding="utf-8") as bf:
        for obj in items:
            bf.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="中文语料流式清洗 → JSONL，支持按清洗后字节/条数试水。"
    )
    p.add_argument("--input_path", required=True, help="输入文件或目录")
    p.add_argument("--output_path", required=True, help="清洗后 JSONL 输出路径")
    p.add_argument("--badcase_path", required=True, help="被过滤样本 JSONL 路径")
    p.add_argument("--stats_path", required=True, help="统计信息 JSON 路径")
    p.add_argument(
        "--input_format",
        choices=("txt", "jsonl"),
        required=True,
        help="输入格式",
    )
    p.add_argument(
        "--text_key",
        default="text",
        help="jsonl 中文本字段名，默认 text",
    )
    p.add_argument("--min_length", type=int, default=10)
    p.add_argument("--max_length", type=int, default=2000)
    p.add_argument("--min_zh_ratio", type=float, default=0.6)
    p.add_argument(
        "--dedup_mode",
        choices=("md5", "prefix"),
        default="md5",
    )
    p.add_argument("--prefix_len", type=int, default=50)
    p.add_argument("--max_badcases", type=int, default=200)
    p.add_argument(
        "--sample_mode",
        choices=("none", "lines", "bytes"),
        default="none",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="sample_mode=lines 时，清洗后保留条数上限",
    )
    p.add_argument(
        "--max_bytes",
        type=int,
        default=None,
        help="sample_mode=bytes 时，清洗后写出 UTF-8 字节上限（如 1073741824 ≈ 1GiB）",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _validate_sample_args(args: argparse.Namespace) -> None:
    if args.sample_mode == "lines":
        if args.max_samples is None or args.max_samples <= 0:
            raise SystemExit(
                "sample_mode=lines 时必须设置正整数 --max_samples"
            )
    if args.sample_mode == "bytes":
        if args.max_bytes is None or args.max_bytes <= 0:
            raise SystemExit(
                "sample_mode=bytes 时必须设置正整数 --max_bytes"
            )


def main() -> None:
    _setup_logging()
    args = parse_args()
    _validate_sample_args(args)

    rng = random.Random(args.seed)
    dedup_set: Set[str] = set()

    stats: Dict[str, Any] = {
        "raw_records": 0,
        "kept": 0,
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "low_zh_ratio": 0,
        "duplicate": 0,
        "output_bytes": 0,
        "stop_reason": "completed_all_input",
    }

    # 全局 badcase 蓄水池（所有过滤原因合计最多 max_badcases 条）
    bad_reservoir: List[Dict[str, str]] = []
    bad_total_seen = 0

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stop_all = False
    pbar = tqdm(desc="处理原始样本", unit="条", dynamic_ncols=True)

    try:
        with open(args.output_path, "w", encoding="utf-8") as out_f:
            for fp in iter_input_files(args.input_path, args.input_format):
                if stop_all:
                    break
                try:
                    for _path_str, _lineno, raw in iter_lines(
                        fp, args.input_format, args.text_key
                    ):
                        if stop_all:
                            break

                        stats["raw_records"] += 1
                        pbar.update(1)

                        cleaned = clean_text(raw)
                        if not cleaned:
                            stats["empty"] += 1
                            bad_total_seen += 1
                            reservoir_offer_badcase(
                                bad_reservoir,
                                {"text": raw[:2000], "reason": "empty"},
                                args.max_badcases,
                                rng,
                                bad_total_seen,
                            )
                            continue

                        ok, reason = should_keep_text(
                            cleaned,
                            args.min_length,
                            args.max_length,
                            args.min_zh_ratio,
                        )
                        if not ok:
                            assert reason is not None
                            stats[reason] += 1
                            bad_total_seen += 1
                            reservoir_offer_badcase(
                                bad_reservoir,
                                {
                                    "text": cleaned[:2000],
                                    "reason": reason,
                                },
                                args.max_badcases,
                                rng,
                                bad_total_seen,
                            )
                            continue

                        sig = make_signature(
                            args.dedup_mode, cleaned, args.prefix_len
                        )
                        if sig in dedup_set:
                            stats["duplicate"] += 1
                            bad_total_seen += 1
                            reservoir_offer_badcase(
                                bad_reservoir,
                                {
                                    "text": cleaned[:2000],
                                    "reason": "duplicate",
                                },
                                args.max_badcases,
                                rng,
                                bad_total_seen,
                            )
                            continue
                        dedup_set.add(sig)

                        line = json.dumps({"text": cleaned}, ensure_ascii=False) + "\n"
                        b = line.encode("utf-8")
                        out_f.write(line)
                        stats["output_bytes"] += len(b)
                        stats["kept"] += 1

                        if args.sample_mode == "lines" and args.max_samples is not None:
                            if stats["kept"] >= args.max_samples:
                                stats["stop_reason"] = "max_samples_reached"
                                stop_all = True
                                break

                        if args.sample_mode == "bytes" and args.max_bytes is not None:
                            if stats["output_bytes"] >= args.max_bytes:
                                stats["stop_reason"] = "max_bytes_reached"
                                stop_all = True
                                break

                except OSError as e:
                    logging.warning("处理文件时出错 %s: %s", fp, e)
                    continue
    finally:
        pbar.close()

    write_badcase(args.badcase_path, bad_reservoir)

    raw_n = stats["raw_records"]
    kept = stats["kept"]
    ratio = (kept / raw_n) if raw_n else 0.0
    stats["keep_ratio"] = round(ratio, 6)
    stats["dedup_mode"] = args.dedup_mode
    stats["sample_mode"] = args.sample_mode
    if args.sample_mode == "lines":
        stats["max_samples"] = args.max_samples
    if args.sample_mode == "bytes":
        stats["max_bytes"] = args.max_bytes

    stats_dir = os.path.dirname(os.path.abspath(args.stats_path))
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    with open(args.stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    # 终端摘要
    print("========== 清洗统计 ==========", file=sys.stderr)
    print(f"原始总条数:     {raw_n}", file=sys.stderr)
    print(f"保留条数:       {kept}", file=sys.stderr)
    print(f"保留比例:       {stats['keep_ratio']}", file=sys.stderr)
    print(f"empty:          {stats['empty']}", file=sys.stderr)
    print(f"too_short:      {stats['too_short']}", file=sys.stderr)
    print(f"too_long:       {stats['too_long']}", file=sys.stderr)
    print(f"low_zh_ratio:   {stats['low_zh_ratio']}", file=sys.stderr)
    print(f"duplicate:      {stats['duplicate']}", file=sys.stderr)
    print(f"输出字节数:     {stats['output_bytes']}", file=sys.stderr)
    print(f"stop_reason:    {stats['stop_reason']}", file=sys.stderr)
    print(f"badcase 保存:   {args.badcase_path} ({len(bad_reservoir)} 条)", file=sys.stderr)
    print("================================", file=sys.stderr)


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()
