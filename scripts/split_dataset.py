#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 JSONL 语料按条数切分为 train / valid / test，可复现打乱。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import List, Tuple

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSONL 语料 train/valid/test 切分")
    p.add_argument("--input_path", required=True, help="输入 jsonl")
    p.add_argument("--train_path", required=True)
    p.add_argument("--valid_path", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--stats_path", required=True, help="统计信息 json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.98)
    p.add_argument("--valid_ratio", type=float, default=0.01)
    p.add_argument("--test_ratio", type=float, default=0.01)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--shuffle", dest="shuffle", action="store_true", help="打乱后再切分（默认）")
    g.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="按文件顺序切分")
    p.set_defaults(shuffle=True)
    return p.parse_args()


def validate_ratios(a: float, b: float, c: float) -> None:
    s = a + b + c
    if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(
            f"比例之和必须为 1（允许绝对误差 1e-6），当前为 {s}。"
            f" train={a} valid={b} test={c}"
        )
    if a < 0 or b < 0 or c < 0:
        raise SystemExit("比例不能为负数。")
    if a == 0:
        raise SystemExit("train_ratio 不能为 0。")


def load_jsonl_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise SystemExit(f"输入文件不存在: {path}")

    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(tqdm(f, desc="读取行", unit="行"), start=1):
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON 解析失败 {path}:{i} — {e}") from e
            lines.append(line)
    return lines


def split_counts(n: int, rt: float, rv: float, rte: float) -> Tuple[int, int, int]:
    """按条数切分；余数归入 test，保证三者之和为 n。"""
    n_train = int(n * rt)
    n_valid = int(n * rv)
    n_test = n - n_train - n_valid
    if n_test < 0:
        raise SystemExit(
            f"切分溢出: n={n} n_train={n_train} n_valid={n_valid}，请检查比例。"
        )
    return n_train, n_valid, n_test


def write_lines(path: str, rows: List[str]) -> int:
    """写入 UTF-8，返回字节数（与磁盘文件一致，每行末尾 \\n）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    total = 0
    with open(path, "w", encoding="utf-8") as out:
        for line in tqdm(rows, desc=f"写入 {os.path.basename(path)}", unit="行"):
            chunk = line + "\n"
            out.write(chunk)
            total += len(chunk.encode("utf-8"))
    return total


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    lines = load_jsonl_lines(args.input_path)
    n = len(lines)
    if n == 0:
        raise SystemExit("输入中没有有效样本行。")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(lines)

    n_train, n_valid, n_test = split_counts(
        n, args.train_ratio, args.valid_ratio, args.test_ratio
    )
    train_rows = lines[:n_train]
    valid_rows = lines[n_train : n_train + n_valid]
    test_rows = lines[n_train + n_valid :]

    train_b = write_lines(args.train_path, train_rows)
    valid_b = write_lines(args.valid_path, valid_rows)
    test_b = write_lines(args.test_path, test_rows)

    stats = {
        "raw_records": n,
        "train_records": len(train_rows),
        "valid_records": len(valid_rows),
        "test_records": len(test_rows),
        "train_ratio_actual": round(len(train_rows) / n, 8),
        "valid_ratio_actual": round(len(valid_rows) / n, 8),
        "test_ratio_actual": round(len(test_rows) / n, 8),
        "train_bytes": train_b,
        "valid_bytes": valid_b,
        "test_bytes": test_b,
        "seed": args.seed,
        "shuffled": args.shuffle,
        "train_ratio_requested": args.train_ratio,
        "valid_ratio_requested": args.valid_ratio,
        "test_ratio_requested": args.test_ratio,
    }

    sd = os.path.dirname(os.path.abspath(args.stats_path))
    if sd:
        os.makedirs(sd, exist_ok=True)
    with open(args.stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    print("切分完成。", file=sys.stderr)
    print(
        f"train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)} "
        f"(raw={n})",
        file=sys.stderr,
    )
    print(f"统计已写入: {args.stats_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
